# CMRxRecon_insightdcu
Prototype and Submission files for the MICCAI 2023 STACOM workshop (CMRxRecon challenge)

The weights for trained denoising UNET are on drive https://drive.google.com/file/d/1muzBNRrypFeifJGC_TvWD55YrlUgfWQ1/view?usp=sharing This denoising UNET was trained on long axis images only.

Currently combining the coils with rss in reconstruct_mc_image.py after the reconstruction with UNET. 

Please double check the sx and sy dimensiions as when I display a slice with matplotlib, the x- and y-axes are flipped

Please double check the predicted (returned) signal intensity normalization in reconstruct_sc_image and reconstruct_mc_image functions
