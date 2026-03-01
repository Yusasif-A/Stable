import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import torch
import torch.optim as optim
import torch.distributed as dist
import torchaudio
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb

from tqdm import tqdm
from dataclasses import asdict

from datas.dataset import StableDataset, collate_fn, intersperse
from datas.sampler import DistributedBucketSampler
from text import symbols, cleaned_text_to_sequence
from text.yoruba import yoruba_to_ipa
from config import MelConfig, ModelConfig, TrainConfig
from models.model import StableTTS
from utils.audio import LogMelSpectrogram, load_and_resample_audio
from utils.scheduler import get_cosine_schedule_with_warmup
from utils.load import continue_training

# --- Eval audio config ---
EVAL_TEXT = 'ẹkún ọmọ kò jẹ kí ìdílé sùn.'          # short Yoruba test sentence
EVAL_REF_AUDIO = '/teamspace/studios/this_studio/data/wavs/audio_000000.wav'  # reference voice
EVAL_INTERVAL = 10   # save audio every N epochs
EVAL_AUDIO_DIR = './eval_audio'
VOCODER_PATH = './checkpoints/vocoder.pt'

torch.backends.cudnn.benchmark = True

os.makedirs(EVAL_AUDIO_DIR, exist_ok=True)

def load_vocoder(device):
    from vocoders.vocos.models.model import Vocos
    from config import VocosConfig
    vocoder = Vocos(VocosConfig(), MelConfig())
    vocoder.load_state_dict(torch.load(VOCODER_PATH, weights_only=True, map_location='cpu'))
    vocoder.eval()
    return vocoder.to(device)

@torch.inference_mode()
def save_eval_audio(model_module, vocoder, mel_extractor, mel_config, epoch, device):
    """Run inference on a fixed test sentence and save audio to eval_audio/."""
    try:
        phones = yoruba_to_ipa(EVAL_TEXT)
        ids = intersperse(cleaned_text_to_sequence(phones), item=0)
        text = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        text_length = torch.tensor([text.size(-1)], dtype=torch.long, device=device)

        ref = load_and_resample_audio(EVAL_REF_AUDIO, mel_config.sample_rate, device=device)
        ref_mel = mel_extractor(ref)

        model_module.eval()
        mel_out = model_module.synthesise(text, text_length, step=10, temperature=1.0,
                                          ref_mel=ref_mel, length_scale=1.0,
                                          solver='euler', cfg=3.0)['decoder_outputs']
        audio = vocoder(mel_out.to(device)).cpu()
        model_module.train()

        out_path = os.path.join(EVAL_AUDIO_DIR, f'epoch_{epoch:04d}.wav')
        torchaudio.save(out_path, audio.squeeze(0).unsqueeze(0), mel_config.sample_rate)
        wandb.log({'eval_audio': wandb.Audio(out_path, sample_rate=mel_config.sample_rate, caption=f'epoch {epoch}')}, step=epoch)
        print(f'[eval] saved {out_path}')
    except Exception as e:
        print(f'[eval] skipped: {e}')

def _cleanup_old_checkpoints(path, keep=3):
    """Delete old checkpoint/optimizer files, keeping the `keep` most recent."""
    epochs = {}
    for f in os.listdir(path):
        if f.endswith('.pt') and '_' in f:
            name, epoch_str = f.rsplit('_', 1)
            try:
                epoch = int(epoch_str.split('.')[0])
                if name.startswith('checkpoint') or name.startswith('optimizer'):
                    epochs.setdefault(epoch, []).append(os.path.join(path, f))
            except ValueError:
                pass
    for epoch in sorted(epochs.keys())[:-keep]:
        for fpath in epochs[epoch]:
            os.remove(fpath)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("gloo" if os.name == "nt" else "nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def _init_config(model_config: ModelConfig, mel_config: MelConfig, train_config: TrainConfig):
    
    if not os.path.exists(train_config.model_save_path):
        print(f'Creating {train_config.model_save_path}')
        os.makedirs(train_config.model_save_path, exist_ok=True)

def train(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    model_config = ModelConfig()
    mel_config = MelConfig()
    train_config = TrainConfig()
    
    _init_config(model_config, mel_config, train_config)
    
    model = StableTTS(len(symbols), mel_config.n_mels, **asdict(model_config)).to(rank)
    
    model = DDP(model, device_ids=[rank])

    train_dataset = StableDataset(train_config.train_dataset_path, mel_config.hop_length)
    train_sampler = DistributedBucketSampler(train_dataset, train_config.batch_size, [32,300,400,500,600,700,800,900,1000], num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4, pin_memory=True, collate_fn=collate_fn, persistent_workers=True)
    
    if rank == 0:
        writer = SummaryWriter(train_config.log_dir)
        wandb.init(project='stabletts-yoruba', config={
            **asdict(train_config), **asdict(mel_config), **asdict(model_config)
        }, resume='allow')
        vocoder = load_vocoder(rank)
        mel_extractor = LogMelSpectrogram(**asdict(mel_config)).to(rank)

    optimizer = optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(train_config.warmup_steps), num_training_steps=train_config.num_epochs * len(train_dataloader))
    
    # load latest checkpoints if possible
    current_epoch = continue_training(train_config.model_save_path, model, optimizer)

    model.train()
    for epoch in range(current_epoch, train_config.num_epochs):  # loop over the train_dataset multiple times
        train_dataloader.batch_sampler.set_epoch(epoch)
        if rank == 0:
            dataloader = tqdm(train_dataloader)
        else:
            dataloader = train_dataloader
            
        for batch_idx, datas in enumerate(dataloader):
            datas = [data.to(rank, non_blocking=True) for data in datas]
            x, x_lengths, y, y_lengths, z, z_lengths = datas
            optimizer.zero_grad()
            dur_loss, diff_loss, prior_loss, _ = model(x, x_lengths, y, y_lengths, z, z_lengths)
            loss = dur_loss + diff_loss + prior_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if rank == 0 and batch_idx % train_config.log_interval == 0:
                steps = epoch * len(dataloader) + batch_idx
                writer.add_scalar("training/diff_loss", diff_loss.item(), steps)
                writer.add_scalar("training/dur_loss", dur_loss.item(), steps)
                writer.add_scalar("training/prior_loss", prior_loss.item(), steps)
                writer.add_scalar("learning_rate/learning_rate", scheduler.get_last_lr()[0], steps)
                wandb.log({
                    'diff_loss': diff_loss.item(),
                    'dur_loss': dur_loss.item(),
                    'prior_loss': prior_loss.item(),
                    'total_loss': loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0],
                    'epoch': epoch,
                }, step=steps)
            
        if rank == 0 and epoch % train_config.save_interval == 0:
            torch.save(model.module.state_dict(), os.path.join(train_config.model_save_path, f'checkpoint_{epoch}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(train_config.model_save_path, f'optimizer_{epoch}.pt'))
            # Keep only the 3 most recent checkpoint pairs to save disk space
            _cleanup_old_checkpoints(train_config.model_save_path, keep=3)
            if epoch % EVAL_INTERVAL == 0:
                save_eval_audio(model.module, vocoder, mel_extractor, mel_config, epoch, rank)
        print(f"Rank {rank}, Epoch {epoch}, Loss {loss.item()}")

    cleanup()
    
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)