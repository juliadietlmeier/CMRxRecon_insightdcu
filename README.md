# CMRxRecon_insightdcu
Prototype and Submission files for the MICCAI 2023 STACOM workshop (CMRxRecon challenge)

The following weights are needed ([Google link](https://drive.google.com/drive/folders/1WnnJEoyEY7-4rcavP4rd02fRPgIJy_1-?usp=drive_link)) to reconstruct Single Coil (SC) and Multi Coil (MC) LAX and SAX images:

SC_GNAUnet_300epochs_lax.pt

SC_GNAUnet_300epochs_sax.pt

MC_GNAUnet_300epochs_lax.pt

MC_GNAUnet_300epochs_sax.pt


The weights for trained denoising prototype UNET are on drive https://drive.google.com/file/d/1muzBNRrypFeifJGC_TvWD55YrlUgfWQ1/view?usp=sharing This denoising UNET was trained on long axis images only.


**Our pipeline processes LAX and SAX images separately**


![architecture_v4](https://github.com/juliadietlmeier/CMRxRecon_insightdcu/assets/79544193/f6f404c8-803c-43eb-b8f1-881389af89f5)
 
