import torch 
from tqdm import tqdm 
import diffusers 
from rho_diffusion.diffusion.abstract_diffusion import AbstractDiffusionPipeline
from rho_diffusion.registry import registry
from typing import Any, Union
from rho_diffusion.diffusion import schedule
from torch import nn, Tensor
from collections.abc import Mapping, Iterable
from rho_diffusion.utils import save_model_checkpoint, sample_from_discrete_parameter_space, right_pad_dims_to
from diffusers.schedulers.scheduling_utils import SchedulerMixin as DiffusersBaseScheduler 

class DiffusersDDPMPipeline(AbstractDiffusionPipeline):
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        backbone: Union[str, type[nn.Module]],
        backbone_kwargs: dict[str, Any],
        schedule: DiffusersBaseScheduler,
        loss_func:Union[str, type[nn.Module], nn.Module],
        timesteps: Union[int, Tensor] = 1000,
        cond_fn: str = None,
        cond_fn_kwargs: dict = None,
        optimizer: Union[str, type[nn.Module]] = None,
        opt_kwargs: Union[Mapping[str, Any], None] = {},
        t_checkpoints=None,
        sampling_batch_size=10,
        sample_every_n_epochs=5,
        sample_parameter_space=None,
        save_checkpoint_every_n_epochs=10,
    ):
        super().__init__(
            backbone=backbone,
            backbone_kwargs=backbone_kwargs,
            schedule=schedule,
            timesteps=timesteps,
            cond_fn=cond_fn,
            cond_fn_kwargs=cond_fn_kwargs,
            optimizer=optimizer,
            opt_kwargs=opt_kwargs
        )
        if isinstance(loss_func, str):
            loss_func = registry.get("nn", loss_func)
        if isinstance(loss_func, type):
            loss_func = loss_func()
        self.loss_func = loss_func
        self.t_checkpoints = t_checkpoints
        self.sampling_batch_size = sampling_batch_size
        self.sample_every_n_epochs = sample_every_n_epochs
        self.sample_parameter_space = sample_parameter_space
        self.save_weights_every_n_epochs = save_checkpoint_every_n_epochs



    
    def training_step(self, batch: Iterable[Any]) -> float:
        """
        Given data, the training objective is to train the backbone
        to be epsilon-theta, i.e. supervised training to predict
        noise.

        Parameters
        ----------
        batch : Iterable[Any]
            Batched data and label pairs, either as a list or
            dictionary.

        Returns
        -------
        float
            Value of the loss to perform backprop with
        """
        if isinstance(batch, list):
            # assume it's a pair of image and labels like MNIST
            data, labels = batch
        elif isinstance(batch, dict):
            data = batch.get("data")
            labels = batch.get("label")
        else:
            # no label
            data = batch
            labels = None
        self.data_shape = data.shape
        self.data_dtype = data.dtype
        # for training, we learn via the forward process
        batch_size = data.size(0)
        t = self.random_timesteps(batch_size).to(data.device)

        

        # if x_data.isnan().sum() > 0:
        #     print("Error: Noised data contains NaNs. Check your noise scheduler.")
        #     import sys
        #     sys.exit(0)

        clean_images = data 

        batch_size = clean_images.shape[0]

        # Sample a set of random time steps for each image in mini-batch
        # timesteps = torch.randint(
        #     0, noise_scheduler.num_train_timesteps, (batch_size,), device=clean_images.device)

        # noisy_images=noise_scheduler.add_noise(clean_images, noise, timesteps)
        noisy_images, noise = self.forward_process(data, t)

        # noise_pred = self.backbone(noisy_images, t)["sample"]
        noise_pred = self.backbone(noisy_images, t, labels)
        # loss = torch.nn.functional.mse_loss(noise_pred, noise)
        if self.schedule.config.prediction_type == 'epsilon':
            loss = self.loss_func(noise_pred, noise)
        elif self.schedule.config.prediction_type == 'sample':
            loss = self.loss_func(noise_pred, noisy_images)
        else:
            raise Exception('Loss cannot be computed because the prediction type is not understood.')
        # accelerator.backward(loss)

        # accelerator.clip_grad_norm_(model.parameters(),1.0)
        torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=1.0, norm_type=2)
        # optimizer.step()
        # lr_scheduler.step()
        # optimizer.zero_grad()

        # progress_bar.update(1)
        # logs = {
        #     "loss" : loss.detach().item(),
        #     "lr" : lr_scheduler.get_last_lr()[0],
        # }
        self.log(f"train_loss", loss, prog_bar=True)
        # progress_bar.set_postfix(**logs)
        return loss

    def forward_process(self, clean_images: Tensor, t: Union[Tensor,  None] = None,) -> list[Tensor]:
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        noisy_images = self.schedule.add_noise(clean_images, noise, t)
        return noisy_images, noise 


    @torch.inference_mode()
    def reverse_process(
        self,
        x_T: Tensor,
        conditions: Tensor = None,
        t_checkpoints: Union[list, Tensor] = None,
    ) -> dict[str, Tensor]:
        
        denoise_steps = max(self.schedule.timesteps).numpy()
        x_t = torch.randn_like(x_T)
        steps = torch.arange(denoise_steps - 1, -1, -1)

        if t_checkpoints is not None:
            num_checkpoints = len(t_checkpoints)
            # make a dummy tensor to get it into the right shape
            dummy = torch.ones_like(x_T, dtype=x_T.dtype).unsqueeze(1)
            dummy = torch.repeat_interleave(dummy, num_checkpoints, 1)
            # dummy serves as a template for the right shape
            denoised_img_buff = torch.zeros_like(dummy)
            steps_per_ckpt = denoise_steps // num_checkpoints
        else:
            denoised_img_buff = None
            steps_per_ckpt = denoise_steps

        if conditions is not None:
            if isinstance(conditions, int):
                cc = torch.full(
                    (batch_size,),
                    fill_value=conditions,
                    device=x_t.device,
                    dtype=torch.long,
                )
            elif isinstance(conditions, str) and conditions == "auto":
                cc = torch.randint(0, 10, (batch_size,), device=x_T.device).long()
            elif isinstance(conditions, torch.Tensor):
                cc = conditions
            elif isinstance(conditions, list):
                cc = torch.tensor(conditions).to(x_t.device)
        else:
            cc = None

            
        # iterate through timesteps
        t_idx = 0
        if self.global_rank == 0:
            steps = tqdm(steps, desc="Reverse diffusion process...")
        for t in steps:
            batch_size = x_T.size(0)
            tt = torch.tensor([t] * batch_size, device=x_T.device)
            with torch.no_grad():
                # out = self.ddim_sample(
                #     self.backbone,
                #     x_t,
                #     tt,
                #     clip_denoised=True,
                #     denoised_fn=None,
                #     cond_fn=None,
                #     model_kwargs={'y': cc},
                # )
                # x_t = out["sample"]

                out = self.backbone(x_t, tt, cc)
                x_t = self.schedule.step(out, int(t), x_t, generator=None)['prev_sample']

            # add image to buffer
            if (
                denoised_img_buff is not None
                and t % steps_per_ckpt == 0
                and t_idx < num_checkpoints
            ):
                denoised_img_buff[:, t_idx] = x_t 
                t_idx += 1

        return {"buffer": denoised_img_buff, "denoised": x_t}



    def generate(self, parameter_space=None, random=False, save_figure_as=None):
        """
        Generate data from noise by sampling from the model.
        """
        if hasattr(self, "data_shape"):
            # infer the data size from `self.data`
            sampling_data_shape = [int(x) for x in self.data_shape]
            sampling_data_shape[0] = self.sampling_batch_size
        else:
            # Construct the shape for inference
            bs = self.sampling_batch_size
            channels = self.backbone_kwargs["out_channels"]
            shape = self.backbone_kwargs["data_shape"]
            sampling_data_shape = torch.Size([bs, channels] + shape)
            self.data_dtype = torch.float32

        sample_data = torch.zeros(
            sampling_data_shape,
            dtype=self.data_dtype,
            device=self.device,
        )

        if parameter_space is None:
            parameter_space = self.sample_parameter_space
        # parameter_space dict --> MultiEmbeddings
        # cond = self.backbone.cond_fn.transform_to_categorical(parameter_space)
        cond = sample_from_discrete_parameter_space(parameter_space, sample_data.shape[0], random=random, device=self.device)

        results = self.reverse_process(
            x_T=sample_data,
            conditions=cond,
            t_checkpoints=self.t_checkpoints,
        )
        figure = self.make_image_grid(
            results["denoised"],
            filename=save_figure_as
        )
        # step = self.trainer.global_step
        # rank = self.trainer.global_rank
        # self.logger.experiment.log_figure(
        #     self.logger.run_id,
        #     figure,
        #     f"images/train_step{step}_rank{rank}_images.png",
        # )
        return figure

        self.scheduler.set_timesteps(199)
        image=torch.randn((1,1,32,32)).to(model.device)
        num_steps=max(noise_scheduler.timesteps).numpy()

        for t in noise_scheduler.timesteps:
            model_output=unet(image,t)['sample']
            image=scheduler.step(model_output,int(t),image,generator=None)['prev_sample']
            if save_process_dir:
                save_image=torchvision.transforms.ToPILImage()(image.squeeze(0))
                save_image.resize((256,256)).save(
                    os.path.join(save_process_dir,"seed-"+str(seed)+"_"+f"{num_steps-t.numpy():03d}"+".png"),format="png")

        return torchvision.transforms.ToPILImage()(image.squeeze(0))
    
    def on_train_epoch_end(self) -> None:
        """
        Check if there is a need to run certain tasks.
        """
        if (
            self.current_epoch > 0 and self.sample_every_n_epochs > 0 
            and self.current_epoch % self.sample_every_n_epochs == 0
        ):
            self.eval()
            self.generate(save_figure_as="output_%d.png" % self.current_epoch)

        if (
            self.current_epoch > 0 and self.save_weights_every_n_epochs > 0
            and self.current_epoch % self.save_weights_every_n_epochs == 0
        ):
            self.eval()
            self.save_model_weights()

    def save_model_weights(self):
        print("saving model checkpoints...")
        save_model_checkpoint(self.backbone, "model.pth")