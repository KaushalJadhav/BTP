import wandb
import os


def init_wandb(cfg,api_key = "9cf3633e3fca66ea64779e02b56d74fb0b046261") -> None:
    """
    Initialize project on Weights & Biases
    Args:
       
    """
    if api_key is not None:
        os.environ['WANDB_API_KEY']=api_key
    wandb.login()

    wandb.init(
        name=cfg.WANDB.NAME,
        config=cfg,
        project=cfg.WANDB.PROJECT,
        resume="allow",
        id=cfg.WANDB.RESTORE_NAME
    )


def wandb_log(train_loss, lr, iter):
    """
    Logs the accuracy and loss to wandb
    Args:
        train_loss (float): Training loss
        val_loss (float): Validation loss
        train_acc (float): Training Accuracy
        val_acc (float): Validation Accuracy
        epoch (int): Epoch Number
    """

    wandb.log({
        'Loss': train_loss,
        'Learning Rate': lr,
    }, step=iter)

def wandb_log_seg(train_loss, lr, it):
    wandb.log({
        'Train Loss': train_loss,
        'Learning Rate': lr,
        'Train Iteration': it
    })

def wandb_log_NAL(loss,lr,it):
    wandb.log({
        'Total Loss':loss[0],
        'CE Loss':loss[1],
        'WCE Loss':loss[2],
        'Learning Rate': lr
    },step=it)