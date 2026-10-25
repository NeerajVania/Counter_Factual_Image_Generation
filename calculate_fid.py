import os
import argparse

import torch
from cleanfid import fid

PATH_REALIMAGES = ""
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate FID scores for generated images.")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., 'basestruct').")
    parser.add_argument("--guide_w", type=float, default=4.0, help="Guidance weight used during generation.")
    args = parser.parse_args()
    guide_w = args.guide_w

    fid_scores = {}
    classes = sorted(os.listdir("out/fid_data/real"))

    for cls in classes:
        print(f"Computing FID for class {cls}")
        real_path = os.path.join(PATH_REALIMAGES, cls)
        
        if not os.path.exists(real_path) or len(os.listdir(real_path)) == 0:
            print(f"Real images for class {cls} not found, skipping FID computation.")
            continue
        
        gen_path = os.path.join(f"out/generated_{args.model}_gw{guide_w}", cls)
        print(gen_path)

        if os.path.exists(gen_path) and len(os.listdir(gen_path)) > 0:
            score = fid.compute_fid(real_path, gen_path)
            fid_scores[cls] = score

    exp_path = f"model_weights/diffusion/{args.model}"
    save_path = exp_path + f"/samples_gw{args.guide_w}" + "/fid_scores.pth"
    torch.save(fid_scores, save_path)
