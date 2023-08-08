# CMRxRecon_insightdcu
Prototype and Submission files for the MICCAI 2023 STACOM workshop (CMRxRecon challenge)

The weights for trained denoising UNET are on drive https://drive.google.com/file/d/1muzBNRrypFeifJGC_TvWD55YrlUgfWQ1/view?usp=sharing This denoising UNET was trained on long axis images only.


Please double check the predicted (returned) signal intensity normalization in reconstruct_sc_image and reconstruct_mc_image functions

**Currently combining the coils with rss in reconstruct_mc_image.py after the reconstruction with UNET** Need to check if the rss or rss_complex should be used in data preparation files in data_utils/ folder in order to squeeze the middle 5th sc dimension ([t,sz,sc,sy,sx]) before slicing 

**Our pipeline processes LAX and SAX images separately**


![architecture_v4](https://github.com/juliadietlmeier/CMRxRecon_insightdcu/assets/79544193/f6f404c8-803c-43eb-b8f1-881389af89f5)
