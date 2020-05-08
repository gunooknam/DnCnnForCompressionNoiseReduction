from PIL import Image
from utils import *
# encoding script
# dataset Generation
# Class D

exePath = "C:\\Users\\gunooknam\\Desktop\\HM16.20\\bin\\vc2015\\x64\\Release\\TAppEncoder.exe "
input_Sequence_folder = "D:\\Test_Sequence\\Class_D_(WQVGA_416x240)"
Select_image_Folder = "D:\\Test_Sequence\\SelectData_qp_32"
width = 416
height = 240
qp = 32
skipFrameCnt = 20

slist = [ os.path.join(input_Sequence_folder,l) for l in os.listdir(input_Sequence_folder) ]
SeqNums=len(slist)
split_rate = 0.75
trainList = slist[ :  int(SeqNums * split_rate)]
testList  = slist[ int(SeqNums * split_rate) : ]

def getFrameCnt(fname) :
    filesize = os.path.getsize(fname)
    frame_bytes = width * height * 3// 2
    return filesize // frame_bytes

def readYframe(fd):
    Y_buf = fd.read(width * height)
    Y = np.reshape(np.frombuffer(Y_buf, dtype=np.uint8), [height, width])
    UV_buf = fd.read(width * height // 2) # skip UV
    return Y


print(" ===== create gt, comp folder ===== ")
for d in [ os.path.join( os.path.join(Select_image_Folder,"gt"), t) for t in ['train', 'test']]:
        makedir(d)
for d in [ os.path.join( os.path.join(Select_image_Folder,"comp"), t) for t in ['train', 'test']]:
        makedir(d)

print(" ===== train data generation ===== ")
for t in trainList:
    basename = os.path.basename(t).split('.')[0]
    seqName = os.path.basename(t).split('_')[0]
    seqCfg = os.path.join("cfg\\per-sequence", seqName +".cfg")
    ctc = "-c cfg\\encoder_intra_main.cfg"
    seqCfgcmd = " -c "+seqCfg
    inp =   " -i "+ t
    oup =   os.path.join("D:\\Test_Sequence\\train",basename +"_recon.yuv")
    bout =  " -b "+ basename +".bin"
    reout = " -o "+ oup
    qpstr = " -q " +str(qp)
    os.system(exePath+ctc+seqCfgcmd+inp+bout+reout+qpstr)
    print("generate compress image ", oup)
    fd_ori = open (t, 'rb')
    fd = open(oup, 'rb')
    n_frame = getFrameCnt(oup)
    for i_frame in range(n_frame):
        if i_frame % skipFrameCnt == 0:
            Y_ori = readYframe(fd_ori)
            Y_comp = readYframe(fd)
            im = Image.fromarray(Y_ori)
            im.save(os.path.join(os.path.join(os.path.join(Select_image_Folder,"gt"),"train"), basename+'_'+str(i_frame)+".png" ) )
            im = Image.fromarray(Y_comp)
            im.save(os.path.join(os.path.join(os.path.join(Select_image_Folder,"comp"),"train"), basename+'_'+str(i_frame)+".png" ) )
        else :
            _ = readYframe(fd_ori)
            _ = readYframe(fd)

print("\n\n ===== test data generation ===== ")
for t in testList:
    basename = os.path.basename(t).split('.')[0]
    seqName = os.path.basename(t).split('_')[0]
    seqCfg = os.path.join("cfg\\per-sequence", seqName +".cfg")
    ctc = "-c cfg\\encoder_intra_main.cfg"
    seqCfgcmd = " -c "+seqCfg
    inp =   " -i "+ t
    oup =   os.path.join("D:\\Test_Sequence\\test",basename +"_recon.yuv")
    bout =  " -b "+ basename +".bin"
    reout = " -o "+ oup
    qpstr = " -q " +str(qp)
    os.system(exePath+ctc+seqCfgcmd+inp+bout+reout+qpstr)
    print("generate compress image ", oup)
    fd_ori = open (t, 'rb')
    fd = open(oup, 'rb')
    n_frame = getFrameCnt(oup)
    for i_frame in range(n_frame):
        if i_frame % skipFrameCnt == 0:
            Y_ori = readYframe(fd_ori)
            Y_comp = readYframe(fd)
            im = Image.fromarray(Y_ori)
            im.save(os.path.join(os.path.join(os.path.join(Select_image_Folder,"gt"),"test"), basename+'_'+str(i_frame)+".png" ) )
            im = Image.fromarray(Y_comp)
            im.save(os.path.join(os.path.join(os.path.join(Select_image_Folder,"comp"),"test"), basename+'_'+str(i_frame)+".png" ) )
        else :
            _ = readYframe(fd_ori)
            _ = readYframe(fd)

