import segmentation_models as sm

from keras.models import Model
from keras.layers import (Add, BatchNormalization, Concatenate, Conv2D, Input,
                          MaxPooling2D, ReLU, UpSampling2D)

def Siamese_FPN():    

    input_size = [256, 256, 3]
    
    BACKBONE = 'vgg19'
    
    model_1_Input = Input(shape = (256, 256, 3), name = 'input_1')
    model_2_Input = Input(shape = (256, 256, 3), name = 'input_2')

    model_1 = sm.FPN(BACKBONE, classes = 1, activation = 'sigmoid', encoder_weights = None, input_shape = (input_size[0], input_size[1], input_size[2]))
    model_2 = sm.FPN(BACKBONE, classes = 1, activation = 'sigmoid', encoder_weights = None, input_shape = (input_size[0], input_size[1], input_size[2]))

    # Encoder_1

    # Block_1
    
    E1_block1_conv1 = model_1.get_layer('block1_conv1')(model_1_Input)
    E1_block1_conv2 = model_1.get_layer('block1_conv2')(E1_block1_conv1)
    E1_block1_pool = model_1.get_layer('block1_pool')(E1_block1_conv2)
    
    # Block_2
    
    E1_block2_conv1 = model_1.get_layer('block2_conv1')(E1_block1_pool)
    E1_block2_conv2 = model_1.get_layer('block2_conv2')(E1_block2_conv1)
    E1_block2_pool = model_1.get_layer('block2_pool')(E1_block2_conv2)
    
    # Block_3
    
    E1_block3_conv1 = model_1.get_layer('block3_conv1')(E1_block2_pool)
    E1_block3_conv2 = model_1.get_layer('block3_conv2')(E1_block3_conv1)
    E1_block3_conv3 = model_1.get_layer('block3_conv3')(E1_block3_conv2)
    E1_block3_conv4 = model_1.get_layer('block3_conv4')(E1_block3_conv3)
    E1_block3_pool = model_1.get_layer('block3_pool')(E1_block3_conv4)
    
    # Block_4
    
    E1_block4_conv1 = model_1.get_layer('block4_conv1')(E1_block3_pool)
    E1_block4_conv2 = model_1.get_layer('block4_conv2')(E1_block4_conv1)
    E1_block4_conv3 = model_1.get_layer('block4_conv3')(E1_block4_conv2)
    E1_block4_conv4 = model_1.get_layer('block4_conv4')(E1_block4_conv3)
    E1_block4_pool = model_1.get_layer('block4_pool')(E1_block4_conv4)
    
    # Block_5
    
    E1_block5_conv1 = model_1.get_layer('block5_conv1')(E1_block4_pool)
    E1_block5_conv2 = model_1.get_layer('block5_conv2')(E1_block5_conv1)
    E1_block5_conv3 = model_1.get_layer('block5_conv3')(E1_block5_conv2)
    E1_block5_conv4 = model_1.get_layer('block5_conv4')(E1_block5_conv3)
    E1_block5_pool = model_1.get_layer('block5_pool')(E1_block5_conv4)
    
    # FPN_P5
    
    E1_fpn_stage_p5_pre_conv = model_1.get_layer('fpn_stage_p5_pre_conv')(E1_block5_pool)
    E1_fpn_stage_p5_upsampling = model_1.get_layer('fpn_stage_p5_upsampling')(E1_fpn_stage_p5_pre_conv)
    E1_fpn_stage_p5_conv = model_1.get_layer('fpn_stage_p5_conv')(E1_block5_conv4)
    E1_fpn_stage_p5_add = model_1.get_layer('fpn_stage_p5_add')([E1_fpn_stage_p5_upsampling, E1_fpn_stage_p5_conv])
    
    # FPN_P4
    
    E1_fpn_stage_p4_upsampling = model_1.get_layer('fpn_stage_p4_upsampling')(E1_fpn_stage_p5_add)
    E1_fpn_stage_p4_conv = model_1.get_layer('fpn_stage_p4_conv')(E1_block4_conv4)
    E1_fpn_stage_p4_add = model_1.get_layer('fpn_stage_p4_add')([E1_fpn_stage_p4_upsampling, E1_fpn_stage_p4_conv])
    
    # FPN_P3
    
    E1_fpn_stage_p3_upsampling = model_1.get_layer('fpn_stage_p3_upsampling')(E1_fpn_stage_p4_add)
    E1_fpn_stage_p3_conv = model_1.get_layer('fpn_stage_p3_conv')(E1_block3_conv4)
    E1_fpn_stage_p3_add = model_1.get_layer('fpn_stage_p3_add')([E1_fpn_stage_p3_upsampling, E1_fpn_stage_p3_conv])
    
    # FPN_P2
    
    E1_fpn_stage_p2_upsampling = model_1.get_layer('fpn_stage_p2_upsampling')(E1_fpn_stage_p3_add)
    E1_fpn_stage_p2_conv = model_1.get_layer('fpn_stage_p2_conv')(E1_block2_conv2)
    E1_fpn_stage_p2_add = model_1.get_layer('fpn_stage_p2_add')([E1_fpn_stage_p2_upsampling, E1_fpn_stage_p2_conv])
    
    # SEGM_3A
    
    E1_segm_stage3a_conv = model_1.get_layer('segm_stage3a_conv')(E1_fpn_stage_p3_add)
    E1_segm_stage3a_bn = model_1.get_layer('segm_stage3a_bn')(E1_segm_stage3a_conv)
    E1_segm_stage3a_relu = model_1.get_layer('segm_stage3a_relu')(E1_segm_stage3a_bn)
    
    # SEGM_4A
    
    E1_segm_stage4a_conv = model_1.get_layer('segm_stage4a_conv')(E1_fpn_stage_p4_add)
    E1_segm_stage4a_bn = model_1.get_layer('segm_stage4a_bn')(E1_segm_stage4a_conv)
    E1_segm_stage4a_relu = model_1.get_layer('segm_stage4a_relu')(E1_segm_stage4a_bn)
    
    # SEGM_5A
    
    E1_segm_stage5a_conv = model_1.get_layer('segm_stage5a_conv')(E1_fpn_stage_p5_add)
    E1_segm_stage5a_bn = model_1.get_layer('segm_stage5a_bn')(E1_segm_stage5a_conv)
    E1_segm_stage5a_relu = model_1.get_layer('segm_stage5a_relu')(E1_segm_stage5a_bn)
    
    # SEGM_2A
    
    E1_segm_stage2a_conv = model_1.get_layer('segm_stage2a_conv')(E1_fpn_stage_p2_add)
    E1_segm_stage2a_bn = model_1.get_layer('segm_stage2a_bn')(E1_segm_stage2a_conv)
    E1_segm_stage2a_relu = model_1.get_layer('segm_stage2a_relu')(E1_segm_stage2a_bn)
    
    # SEGM_3B
    
    E1_segm_stage3b_conv = model_1.get_layer('segm_stage3b_conv')(E1_segm_stage3a_relu)
    E1_segm_stage3b_bn = model_1.get_layer('segm_stage3b_bn')(E1_segm_stage3b_conv)
    E1_segm_stage3b_relu = model_1.get_layer('segm_stage3b_relu')(E1_segm_stage3b_bn)
    
    # SEGM_4B
    
    E1_segm_stage4b_conv = model_1.get_layer('segm_stage4b_conv')(E1_segm_stage4a_relu)
    E1_segm_stage4b_bn = model_1.get_layer('segm_stage4b_bn')(E1_segm_stage4b_conv)
    E1_segm_stage4b_relu = model_1.get_layer('segm_stage4b_relu')(E1_segm_stage4b_bn)
    
    # SEGM_5B
    
    E1_segm_stage5b_conv = model_1.get_layer('segm_stage5b_conv')(E1_segm_stage5a_relu)
    E1_segm_stage5b_bn = model_1.get_layer('segm_stage5b_bn')(E1_segm_stage5b_conv)
    E1_segm_stage5b_relu = model_1.get_layer('segm_stage5b_relu')(E1_segm_stage5b_bn)
    
    # SEGM_2B
    
    E1_segm_stage2b_conv = model_1.get_layer('segm_stage2b_conv')(E1_segm_stage2a_relu)
    E1_segm_stage2b_bn = model_1.get_layer('segm_stage2b_bn')(E1_segm_stage2b_conv)
    E1_segm_stage2b_relu = model_1.get_layer('segm_stage2b_relu')(E1_segm_stage2b_bn)
    
    # Upsampling & Concat
    
    E1_upsampling_stage3 = model_1.get_layer('upsampling_stage3')(E1_segm_stage3b_relu)
    E1_upsampling_stage4 = model_1.get_layer('upsampling_stage4')(E1_segm_stage4b_relu)
    E1_upsampling_stage5 = model_1.get_layer('upsampling_stage5')(E1_segm_stage5b_relu)

    E1_aggregation_concat = model_1.get_layer('aggregation_concat')([E1_segm_stage2b_relu, E1_upsampling_stage3, E1_upsampling_stage4, E1_upsampling_stage5])

    # Encoder_2

    # Block_1
    
    E2_block1_conv1 = model_2.get_layer('block1_conv1')(model_2_Input)
    E2_block1_conv2 = model_2.get_layer('block1_conv2')(E2_block1_conv1)
    E2_block1_pool = model_2.get_layer('block1_pool')(E2_block1_conv2)
    
    # Block_2
    
    E2_block2_conv1 = model_2.get_layer('block2_conv1')(E2_block1_pool)
    E2_block2_conv2 = model_2.get_layer('block2_conv2')(E2_block2_conv1)
    E2_block2_pool = model_2.get_layer('block2_pool')(E2_block2_conv2)
    
    # Block_3
    
    E2_block3_conv1 = model_2.get_layer('block3_conv1')(E2_block2_pool)
    E2_block3_conv2 = model_2.get_layer('block3_conv2')(E2_block3_conv1)
    E2_block3_conv3 = model_2.get_layer('block3_conv3')(E2_block3_conv2)
    E2_block3_conv4 = model_2.get_layer('block3_conv4')(E2_block3_conv3)
    E2_block3_pool = model_2.get_layer('block3_pool')(E2_block3_conv4)
    
    # Block_4
    
    E2_block4_conv1 = model_2.get_layer('block4_conv1')(E2_block3_pool)
    E2_block4_conv2 = model_2.get_layer('block4_conv2')(E2_block4_conv1)
    E2_block4_conv3 = model_2.get_layer('block4_conv3')(E2_block4_conv2)
    E2_block4_conv4 = model_2.get_layer('block4_conv4')(E2_block4_conv3)
    E2_block4_pool = model_2.get_layer('block4_pool')(E2_block4_conv4)
    
    # Block_5
    
    E2_block5_conv1 = model_2.get_layer('block5_conv1')(E2_block4_pool)
    E2_block5_conv2 = model_2.get_layer('block5_conv2')(E2_block5_conv1)
    E2_block5_conv3 = model_2.get_layer('block5_conv3')(E2_block5_conv2)
    E2_block5_conv4 = model_2.get_layer('block5_conv4')(E2_block5_conv3)
    E2_block5_pool = model_2.get_layer('block5_pool')(E2_block5_conv4)
    
    # FPN_P5
    
    E2_fpn_stage_p5_pre_conv = model_2.get_layer('fpn_stage_p5_pre_conv')(E2_block5_pool)
    E2_fpn_stage_p5_upsampling = model_2.get_layer('fpn_stage_p5_upsampling')(E2_fpn_stage_p5_pre_conv)
    E2_fpn_stage_p5_conv = model_2.get_layer('fpn_stage_p5_conv')(E2_block5_conv4)
    E2_fpn_stage_p5_add = model_2.get_layer('fpn_stage_p5_add')([E2_fpn_stage_p5_upsampling, E2_fpn_stage_p5_conv])
    
    # FPN_P4
    
    E2_fpn_stage_p4_upsampling = model_2.get_layer('fpn_stage_p4_upsampling')(E2_fpn_stage_p5_add)
    E2_fpn_stage_p4_conv = model_2.get_layer('fpn_stage_p4_conv')(E2_block4_conv4)
    E2_fpn_stage_p4_add = model_2.get_layer('fpn_stage_p4_add')([E2_fpn_stage_p4_upsampling, E2_fpn_stage_p4_conv])
    
    # FPN_P3
    
    E2_fpn_stage_p3_upsampling = model_2.get_layer('fpn_stage_p3_upsampling')(E2_fpn_stage_p4_add)
    E2_fpn_stage_p3_conv = model_2.get_layer('fpn_stage_p3_conv')(E2_block3_conv4)
    E2_fpn_stage_p3_add = model_2.get_layer('fpn_stage_p3_add')([E2_fpn_stage_p3_upsampling, E2_fpn_stage_p3_conv])
    
    # FPN_P2
    
    E2_fpn_stage_p2_upsampling = model_2.get_layer('fpn_stage_p2_upsampling')(E2_fpn_stage_p3_add)
    E2_fpn_stage_p2_conv = model_2.get_layer('fpn_stage_p2_conv')(E2_block2_conv2)
    E2_fpn_stage_p2_add = model_2.get_layer('fpn_stage_p2_add')([E2_fpn_stage_p2_upsampling, E2_fpn_stage_p2_conv])
    
    # SEGM_3A
    
    E2_segm_stage3a_conv = model_2.get_layer('segm_stage3a_conv')(E2_fpn_stage_p3_add)
    E2_segm_stage3a_bn = model_2.get_layer('segm_stage3a_bn')(E2_segm_stage3a_conv)
    E2_segm_stage3a_relu = model_2.get_layer('segm_stage3a_relu')(E2_segm_stage3a_bn)
    
    # SEGM_4A
    
    E2_segm_stage4a_conv = model_2.get_layer('segm_stage4a_conv')(E2_fpn_stage_p4_add)
    E2_segm_stage4a_bn = model_2.get_layer('segm_stage4a_bn')(E2_segm_stage4a_conv)
    E2_segm_stage4a_relu = model_2.get_layer('segm_stage4a_relu')(E2_segm_stage4a_bn)
    
    # SEGM_5A
    
    E2_segm_stage5a_conv = model_2.get_layer('segm_stage5a_conv')(E2_fpn_stage_p5_add)
    E2_segm_stage5a_bn = model_2.get_layer('segm_stage5a_bn')(E2_segm_stage5a_conv)
    E2_segm_stage5a_relu = model_2.get_layer('segm_stage5a_relu')(E2_segm_stage5a_bn)
    
    # SEGM_2A
    
    E2_segm_stage2a_conv = model_2.get_layer('segm_stage2a_conv')(E2_fpn_stage_p2_add)
    E2_segm_stage2a_bn = model_2.get_layer('segm_stage2a_bn')(E2_segm_stage2a_conv)
    E2_segm_stage2a_relu = model_2.get_layer('segm_stage2a_relu')(E2_segm_stage2a_bn)
    
    # SEGM_3B
    
    E2_segm_stage3b_conv = model_2.get_layer('segm_stage3b_conv')(E2_segm_stage3a_relu)
    E2_segm_stage3b_bn = model_2.get_layer('segm_stage3b_bn')(E2_segm_stage3b_conv)
    E2_segm_stage3b_relu = model_2.get_layer('segm_stage3b_relu')(E2_segm_stage3b_bn)
    
    # SEGM_4B
    
    E2_segm_stage4b_conv = model_2.get_layer('segm_stage4b_conv')(E2_segm_stage4a_relu)
    E2_segm_stage4b_bn = model_2.get_layer('segm_stage4b_bn')(E2_segm_stage4b_conv)
    E2_segm_stage4b_relu = model_2.get_layer('segm_stage4b_relu')(E2_segm_stage4b_bn)
    
    # SEGM_5B
    
    E2_segm_stage5b_conv = model_2.get_layer('segm_stage5b_conv')(E2_segm_stage5a_relu)
    E2_segm_stage5b_bn = model_2.get_layer('segm_stage5b_bn')(E2_segm_stage5b_conv)
    E2_segm_stage5b_relu = model_2.get_layer('segm_stage5b_relu')(E2_segm_stage5b_bn)
    
    # SEGM_2B
    
    E2_segm_stage2b_conv = model_2.get_layer('segm_stage2b_conv')(E2_segm_stage2a_relu)
    E2_segm_stage2b_bn = model_2.get_layer('segm_stage2b_bn')(E2_segm_stage2b_conv)
    E2_segm_stage2b_relu = model_2.get_layer('segm_stage2b_relu')(E2_segm_stage2b_bn)
    
    # Upsampling & Concat
    
    E2_upsampling_stage3 = model_2.get_layer('upsampling_stage3')(E2_segm_stage3b_relu)
    E2_upsampling_stage4 = model_2.get_layer('upsampling_stage4')(E2_segm_stage4b_relu)
    E2_upsampling_stage5 = model_2.get_layer('upsampling_stage5')(E2_segm_stage5b_relu)

    E2_aggregation_concat = model_2.get_layer('aggregation_concat')([E2_segm_stage2b_relu, E2_upsampling_stage3, E2_upsampling_stage4, E2_upsampling_stage5])

    # Final_Concat
    
    E1_Model = Model(inputs = model_1_Input, outputs = E1_aggregation_concat)
    
    for layer in E1_Model.layers:
        layer.name = layer.name + str('_E1')
        
    E2_Model = Model(inputs = model_2_Input, outputs = E2_aggregation_concat)
    
    for layer in E2_Model.layers:
        layer.name = layer.name + str('_E2')
        
    E1_E2_concat = Concatenate(name='E1_E2_concat')([E1_Model.output, E2_Model.output])
    E1_E2_conv = Conv2D(512, (3, 3), padding='same',name='E1_E2_conv')(E1_E2_concat)

    # Decoder
    
    D_final_stage_conv = model_2.get_layer('final_stage_conv')(E1_E2_conv)
    D_final_stage_bn = model_2.get_layer('final_stage_bn')(D_final_stage_conv)
    D_final_stage_relu = model_2.get_layer('final_stage_relu')(D_final_stage_bn)
    D_final_upsampling = model_2.get_layer('final_upsampling')(D_final_stage_relu)
    
    # Final_Stage
    
    D_head_conv = model_2.get_layer('head_conv')(D_final_upsampling)
    D_head_conv = model_2.get_layer('sigmoid')(D_head_conv)
    D_head_conv = Model(inputs = [E1_Model.input, E2_Model.input], outputs = D_head_conv)

    return D_head_conv
