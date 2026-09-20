"""Publication-like augmentation and fixed, checkpoint-local validation."""
import io
import hashlib
import numpy as np
import torch
from PIL import Image


class PublicationAugment:
    """Preserve shape: mild crop OR whole-image padding, then image degradation."""
    def __init__(self, size):
        from torchvision import transforms as T
        self.size = size
        self.crop = T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05))
        self.colour = T.Compose([
            T.ColorJitter(0.25, 0.25, 0.25, 0.04),
            T.RandomGrayscale(p=0.10),
            T.RandomApply([T.GaussianBlur(5, (0.1, 1.2))], p=0.15),
        ])

    def __call__(self, im):
        if torch.rand(()).item() < 0.25:
            im = self.crop(im)
        else:
            # Fit the complete image, with modest translation inside the margins.
            scale = float(torch.empty(()).uniform_(0.72, 1.0))
            w, h = im.size
            factor = self.size * scale / max(w, h)
            small = im.resize((max(1, round(w*factor)), max(1, round(h*factor))), Image.Resampling.BILINEAR)
            corners = np.array([im.getpixel(p) for p in [(0,0),(w-1,0),(0,h-1),(w-1,h-1)]])
            fill = tuple(np.median(corners, axis=0).astype('uint8').tolist())
            canvas = Image.new('RGB', (self.size, self.size), fill)
            x = int(torch.randint(self.size-small.width+1, ()).item())
            y = int(torch.randint(self.size-small.height+1, ()).item())
            canvas.paste(small, (x,y)); im = canvas
        im = self.colour(im)
        if torch.rand(()).item() < 0.30:
            n = int(torch.randint(max(32, self.size//2), self.size+1, ()).item())
            im = im.resize((n,n), Image.Resampling.BILINEAR).resize((self.size,self.size), Image.Resampling.BILINEAR)
        if torch.rand(()).item() < 0.25:
            buf = io.BytesIO()
            im.save(buf, format='JPEG', quality=int(torch.randint(35,96,()).item()))
            buf.seek(0)
            with Image.open(buf) as compressed:
                im = compressed.convert('RGB').copy()
        return im


def validation_panel(man, count, seed):
    val = man[man.split == 'val'].copy()
    good = sorted(set(val.loc[val.view == 0, 'pdb_l']) & set(val.loc[val.view > 0, 'pdb_l']))
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(good, min(count, len(good)), replace=False).tolist())
    panel = val[val.pdb_l.isin(chosen)].sort_values('render_id').reset_index(drop=True)
    if not len(panel):
        raise ValueError('No validation entries with disjoint query/gallery views')
    digest = hashlib.sha256('\n'.join(panel.render_id).encode()).hexdigest()
    return panel, digest


@torch.no_grad()
def validate_checkpoint(model, head, panel, dataset_cls, size, workers, device, rule='top5'):
    from retrieval_scoring import build_pad_index, aggregate
    states = model.training, head.training
    model.eval(); head.eval()
    try:
        ds = dataset_cls(list(zip(panel.shard, panel.member)), size, train=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=workers,
                                             pin_memory=True, shuffle=False)
        chunks = []
        for x in loader:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                f = model(pixel_values=x.to(device)).last_hidden_state[:,0]
            chunks.append(head(f.float()).cpu().numpy())
        emb = np.concatenate(chunks)
        q = (panel.view == 0).to_numpy(); g = (panel.view > 0).to_numpy()
        assert not set(panel.loc[q,'render_id']) & set(panel.loc[g,'render_id'])
        vocab = sorted(panel.pdb_l.unique()); pix = {p:i for i,p in enumerate(vocab)}
        gi = torch.tensor([pix[p] for p in panel.loc[g,'pdb_l']], device=device)
        qi = torch.tensor([pix[p] for p in panel.loc[q,'pdb_l']], device=device)
        pad, mask = build_pad_index(gi, len(vocab), device)
        Q = torch.tensor(emb[q],device=device); G = torch.tensor(emb[g],device=device)
        Q = torch.nn.functional.normalize(Q,dim=1); G = torch.nn.functional.normalize(G,dim=1)
        ranks = []
        for start in range(0,len(Q),64):
            scores = aggregate(Q[start:start+64] @ G.T, pad, mask, rule)
            truth = qi[start:start+64]; ts = scores.gather(1,truth[:,None])
            ix = torch.arange(len(vocab),device=device)[None,:]
            ranks.append(((scores>ts)|((scores==ts)&(ix<truth[:,None]))).sum(1)+1)
        r = torch.cat(ranks).float()
        return {**{f'R@{k}':float((r<=k).float().mean()) for k in (1,5,10)},
                'MRR':float((1/r).mean()),'n_queries':len(Q),'n_gallery_images':len(G)}
    finally:
        model.train(states[0]); head.train(states[1])
