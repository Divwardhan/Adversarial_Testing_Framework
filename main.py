from framework import FrameworkRun
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Min-Max Adversarial Training Framework")

    # 🔴 Dataset
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")

    # 🔴 Model & Generator
    parser.add_argument("--model_type", type=str, default="Unet", help="Model type")
    parser.add_argument("--gen_type", type=str, default="edge", help="Generator type")

    # 🔴 Training params
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=128)

    parser.add_argument("--lr_model", type=float, default=1e-3)
    parser.add_argument("--lr_gen", type=float, default=1e-3)

    parser.add_argument("--pretrain_epochs", type=int, default=5)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--gen_epochs", type=int, default=3)
    parser.add_argument("--model_epochs", type=int, default=3)

    # 🔴 System
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")

    # 🔴 Saving / Logging
    parser.add_argument("--save_images", action="store_true", help="Save adversarial images")
    parser.add_argument("--save_dir", type=str, default="outputs")

    parser.add_argument("--max_buffer_size", type=int, default=5000)

    return parser.parse_args()


def main():
    args = parse_args()

    # 🔴 Device handling
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, switching to CPU")
        args.device = "cpu"
    
    print("\n========== CONFIG ==========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("============================\n")

    # 🔴 Run framework
    model, generator = FrameworkRun(
        dataset_path=args.dataset_path,
        model_type=args.model_type,
        gen_type=args.gen_type,
        device=args.device,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr_model=args.lr_model,
        lr_gen=args.lr_gen,
        pretrain_epochs=args.pretrain_epochs,
        cycles=args.cycles,
        gen_epochs=args.gen_epochs,
        model_epochs=args.model_epochs,
        save_images=args.save_images,
        save_dir=args.save_dir,
        max_buffer_size=args.max_buffer_size
    )

    print("\n[INFO] Run finished successfully.")


if __name__ == "__main__":
    main()