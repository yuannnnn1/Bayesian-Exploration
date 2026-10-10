"""Download Oxford-IIIT Pet and cache frozen pretrained features."""
from __future__ import annotations

import argparse
from pathlib import Path

import open_clip
import torch
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet


def load_clip(device: torch.device, weights="openai"):
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "RN50", pretrained=weights, force_quick_gelu=True)
    clip_model = clip_model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("RN50")
    return clip_model, clip_preprocess, tokenizer


def load_models(device: torch.device, weights="openai"):
    clip_model, clip_preprocess, tokenizer = load_clip(device, weights)
    resnet = torch.hub.load("facebookresearch/dino:main", "dino_resnet50", trust_repo=True)
    resnet = resnet.eval().to(device)
    resnet_preprocess = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return clip_model, clip_preprocess, tokenizer, resnet, resnet_preprocess


@torch.inference_mode()
def cache_split(dataset, clip_model, clip_preprocess, resnet, resnet_preprocess, device, batch_size):
    clip_values, resnet_values, labels = [], [], []
    for start in range(0, len(dataset), batch_size):
        images, batch_labels = zip(*(dataset[i] for i in range(start, min(start + batch_size, len(dataset)))))
        clip_batch = torch.stack([clip_preprocess(image) for image in images]).to(device)
        resnet_batch = torch.stack([resnet_preprocess(image) for image in images]).to(device)
        clip_values.append(torch.nn.functional.normalize(clip_model.encode_image(clip_batch).float(), dim=-1).cpu())
        resnet_values.append(torch.nn.functional.normalize(resnet(resnet_batch).float(), dim=-1).cpu())
        labels.append(torch.tensor(batch_labels))
    return {"clip": torch.cat(clip_values), "resnet": torch.cat(resnet_values), "labels": torch.cat(labels)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--output", default="artifacts/features")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--text-only", action="store_true", help="refresh only CLIP text metadata")
    parser.add_argument("--clip-weights", default="openai", help="pretrained tag or local checkpoint")
    args = parser.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.text_only:
        clip_model, _, tokenizer = load_clip(device, args.clip_weights)
    else:
        clip_model, clip_preprocess, tokenizer, resnet, resnet_preprocess = load_models(device, args.clip_weights)
    for split in ("trainval", "test"):
        dataset = OxfordIIITPet(root, split=split, target_types="category", download=True)
        if split == "trainval":
            text = tokenizer([f"a photo of a {name.lower()}" for name in dataset.classes]).to(device)
            text_features = torch.nn.functional.normalize(clip_model.encode_text(text).float(), dim=-1).cpu()
            torch.save({"classes": dataset.classes, "text_features": text_features,
                        "force_quick_gelu": True, "model": "RN50", "pretrained": "openai",
                        "text_prompt": "a photo of a {lowercase_class_name}"},
                       output / "metadata.pt")
        if args.text_only:
            print(f"refreshed {len(dataset.classes)} class text features with QuickGELU")
            return
        cache = cache_split(dataset, clip_model, clip_preprocess, resnet, resnet_preprocess, device, args.batch_size)
        torch.save(cache, output / f"{split}.pt")
        print(f"cached {split}: {len(dataset)} images")


if __name__ == "__main__":
    main()
