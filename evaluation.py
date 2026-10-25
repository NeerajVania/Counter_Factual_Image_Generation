import os
import argparse
import torch
import torchvision 
import numpy as np
from pathlib import Path
from ori.models.resnet import resnet
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base")
    parser.add_argument("--guide_w", type=float, default=4.0)
    args = parser.parse_args()

    exp_path = f"model_weights/diffusion/{args.model}"
    img_path = f"out/generated_{args.model}_gw{args.guide_w}"

    Path(f"{exp_path}/samples_gw{args.guide_w}").mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    classifier = resnet(num_classes=5).to(device)

    ### LOAD CLASSIFIER CHECKPOINT
    classifier.load_state_dict(torch.load("WEIGHTS.pt", map_location=device))
    ###

    classifier.eval()
    dr_levels = [0.0, 1.0, 2.0, 3.0, 4.0]
    samples = []
    labels = []
    for cls in dr_levels:
        gen_path = os.path.join(f"out/generated_{args.model}_gw{args.guide_w}", str(cls))
        img_files = os.listdir(gen_path)
        for img_file in img_files:
            img = torchvision.io.read_image(os.path.join(gen_path, img_file))
            samples.append(img)
            labels.append(cls)

    v2 = torchvision.transforms.Normalize(
        mean=[0.3704248070716858, 0.2282254546880722, 0.13915641605854034],
        std=[0.23381589353084564, 0.1512117236852646, 0.09653093665838242]
    )

    batch_size = 256
    all_preds = []
    num_samples = len(samples)

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = torch.stack(samples[i:i+batch_size]).float() / 255.0
            batch = v2(batch)
            batch = batch.to(device)
            logits = classifier(batch)
            preds = torch.sum(torch.sigmoid(logits) > 0.5, dim=1).cpu().numpy()
            all_preds.extend(preds)

    print(f"amount of predictions: {len(all_preds)}")

    cm = confusion_matrix(labels, np.array(all_preds), labels=[0, 1, 2, 3, 4])
    torch.save(torch.tensor(cm, dtype=torch.float32), f"{exp_path}/samples_gw{args.guide_w}/confusion_matrix.pth")
    accuracy = np.trace(cm) / np.sum(cm)

    confusion_counts = torch.tensor(cm, dtype=torch.float32)
    total = confusion_counts.sum()
    C = 5
    W = torch.zeros((C, C), dtype=torch.float32)
    for i in range(C):
        for j in range(C):
            W[i, j] = ((i - j) ** 2) / ((C - 1) ** 2)
    O = confusion_counts / total

    row_marginals = confusion_counts.sum(dim=1) / total
    col_marginals = confusion_counts.sum(dim=0) / total
    E = torch.ger(row_marginals, col_marginals)

    numerator = (W * O).sum()
    denominator = (W * E).sum()

    kappa_squared = 1 - numerator / (denominator + 1e-8)

    metrics = {"accuracy": accuracy, "kappa_squared": kappa_squared}
    print(f"Accuracy: {accuracy:.4f}, Kappa Squared: {kappa_squared:.4f}")
    torch.save(metrics, f"{exp_path}/samples_gw{args.guide_w}/metrics.pth")