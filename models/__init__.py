from models.clothes import *
from models.down import *
from models.down1 import *
from models.gender import *
from models.age import *
from models.downcolor import *
from models.upcolor import *
from models.downcolor1 import *
from models.downcolor2 import *
from models.downcolor3 import *
from models.bag import *
from models.bag1 import *
from models.bag2 import *
from models.backpack import *
from models.backpack1 import *
from models.backpack2 import *
from models.dense import *


def get_down(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7), **kwargs):
    return down_densenet201(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,global_pooling_size=gap_size, ) 

def get_down1(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7), **kwargs):
    return down1_densenet201(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,global_pooling_size=gap_size, ) 

def get_bag1(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7), **kwargs):
    return bag1_densenet201(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,global_pooling_size=gap_size, ) 

def get_clothes(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7), **kwargs):
    return clothes_densenet201(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,global_pooling_size=gap_size, )  

def get_bag(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7), **kwargs):
    return bag_densenet201(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,global_pooling_size=gap_size, )  

def get_bag2(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7), **kwargs):
    return bag2_densenet201(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,global_pooling_size=gap_size, )  


def get_backpack(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7), **kwargs):
    return backpack_densenet201(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,global_pooling_size=gap_size, )

def get_backpack1(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7), **kwargs):
    return backpack1_densenet201(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,global_pooling_size=gap_size, )

def get_backpack2(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7), **kwargs):
    return backpack2_densenet201(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,global_pooling_size=gap_size, )

def get_gender(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7), **kwargs):
    return gender_densenet201(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,global_pooling_size=gap_size, )

def get_age(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7), **kwargs):
    return age_densenet201(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,global_pooling_size=gap_size, )   
 
def get_downcolor(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7), **kwargs):
    return downcolor_densenet201(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,global_pooling_size=gap_size, )

def get_upcolor(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7), **kwargs):
    return upcolor_densenet201(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,global_pooling_size=gap_size, )   

def get_downcolor1(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7), **kwargs):
    return downcolor1_densenet201(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,global_pooling_size=gap_size, )   

def get_downcolor2(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7), **kwargs):
    return downcolor2_densenet201(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,global_pooling_size=gap_size, )   

def get_downcolor3(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7), **kwargs):
    return downcolor3_densenet201(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,global_pooling_size=gap_size, )   


def get_dense(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7), **kwargs):
    return densenet201(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,global_pooling_size=gap_size, )


