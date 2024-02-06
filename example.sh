python grounded_sam_demo.py \
    --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --grounded_checkpoint checkpoints/groundingdino_swint_ogc.pth  \
    --sam_checkpoint checkpoints/sam_hq_vit_h.pth --use_sam_hq --sam_version "vit_h" \
    --device "cuda" \
    --input_image /mnt/nas/Projects/Lay3rs/HotelDeLaMarine_2024/session_29_01/Photos/Iphone_15/Salon/paniers_huitres \
    --output_dir /mnt/nas/Projects/Lay3rs/HotelDeLaMarine_2024/session_29_01/Photos/Iphone_15/Salon/paniers_huitres/masks_thrs0.3 \
    --text_prompt "bin with breads." \
    --box_threshold 0.3
