import torch
import important.base.initialization as init

## Triple
    ## SMART-Net
class Multi_Task_Model(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.seg_decoder)
        init.initialize_decoder(self.rec_decoder)

        init.initialize_head(self.segmentation_head)
        init.initialize_head(self.classification_head)
        init.initialize_head(self.reconstruction_head)

    def forward(self, x):
        # feature extract
        feature_list = self.encoder(x)

        # Cls
        labels = self.classification_head(feature_list[-1])

        # Seg
        masks = self.seg_decoder(*feature_list) # 일단은 head 사용 안 함.
        #seg_decoder_output = self.seg_decoder(*feature_list)
        #masks = self.segmentation_head(seg_decoder_output)

        # Rec
        rec_decoder_output = self.rec_decoder(feature_list[-1])
        restores = self.reconstruction_head(rec_decoder_output)

        return labels, masks, restores


## Single
    # CLS
class Single_Task_Model_CLS(torch.nn.Module):

    def initialize(self):
        init.initialize_head(self.classification_head)

    def forward(self, x):
        # feature extract
        feature_list = self.encoder(x)
        
        # Cls
        labels = self.classification_head(feature_list[-1]) 

        return labels

    # SEG
class Single_Task_Model_SEG(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.seg_decoder)

        init.initialize_head(self.segmentation_head)

    def forward(self, x):
        # feature extract
        feature_list = self.encoder(x)

        # Seg
        masks = self.seg_decoder(*feature_list)
        #masks = self.segmentation_head(seg_decoder_output)

        return masks    