import numpy as np
import numpy.linalg as lin 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 

img = []
histor = []
hist_date = []
date2 = []
x_List = []
x_List2 = []
sum_all = 0
bar1 = ''
size = 10
count = 0

def img_read_all(): # 사진 400+a 
    file = open('txt/all.txt','r')
    All=file.read()
    file.close()
    imsi_all = All.split('\n')

    return imsi_all
        
def gradeision_remove(img):
    box_size = 20
    s_box = np.zeros((box_size,box_size))
    b_box = np.zeros((box_size*2,box_size*2))
    Image = img
    
    Image_location = np.argwhere(Image == Image).transpose()
    one_1 = np.full((1,len(Image)*len(Image[0])),1)
    one_1 = one_1[0]
    Image_location2 = np.vstack((Image_location,one_1))

    location = np.array([[],[]])
    s_box_location1 = []
    s_box_location2 = []
    a = np.array([[0,len(Image)-(box_size),0,len(Image)-(box_size)],[0,0,len(Image)-(box_size),len(Image)-(box_size)]])
    b = np.array([[0,20,0,20],[0,0,20,20]])
    one = np.full((1,len(b_box)*len(b_box)),1)
    one = one[0]
    for i in range(0,4):
        s_box_location1.append(np.argwhere(s_box == s_box).transpose())
        s_box_location2.append(np.argwhere(s_box == s_box).transpose())
        s_box_location1[i][0]+=a[0][i]
        s_box_location1[i][1]+=a[1][i]
        s_box_location2[i][0]+=b[0][i]
        s_box_location2[i][1]+=b[1][i]
    for i in range(len(s_box_location1)):
        b_box[s_box_location2[i][0],s_box_location2[i][1]] = Image[s_box_location1[i][0],s_box_location1[i][1]]
        location = np.hstack((location,s_box_location1[i]))
    location = np.vstack((location,one))
    b_box = b_box.reshape(1,len(b_box)*len(b_box[0]))
    b_box = b_box[0]
    abc = np.dot(lin.pinv(location.transpose()),b_box)
    gradeision = (Image_location2[0]*abc[0])+(Image_location2[1]*abc[1])+(Image_location2[2]*abc[2])
    gradeision = gradeision.reshape(len(Image),len(Image[0]))
    Image = Image - gradeision
    Image-=np.min(Image)
    
    return Image

def mser_histo(patch_means):
    global count
    global histor
    Image2 = None
    
    Image2 = patch_means.astype(int)
    if count == 0:
        histor = []
        hist_date = []
        
        for i in range(256):
             histor.append(len(np.argwhere(Image2 == i)))
        hist_date = np.array(histor)
        count = 1
        #print(hist_date)
        #print(len(hist_date))
        
        return hist_date
    
    elif count == 1:
        histor = []
        hist_date = []
        
        for i in range(256):
             histor.append(len(np.argwhere(Image2 == i)))
        hist_date = np.array(histor)
        count = 0
        #print(hist_date)
        #print(len(hist_date))
        
        return hist_date

def mser_histo_see(patch_means,a):
    global count
    global histor
    Image2 = None
    
    Image2 = patch_means.astype(int)
    x_List = np.zeros((200,200))
    for i in range(0,a,1):
        imsi2 = np.where(Image2 == (i))
        x_List[imsi2[0],imsi2[1]] = 255 - i
    return x_List
    
def date_mser(hist_date,sha):
    for i in range(0,len(hist_date)):
        re = (len(hist_date) - (i+1))
        if hist_date[re] == 0:
            continue
        if hist_date[re] >= 1:
            tmax = re+1
            break
    '''
    hold = tmax - rmin #중간 값.(hold)
    num = ((hold) / 2) #나누기.(num)
    ms2 = hold - num # 임계값.(thr)
    '''
    print("max:",tmax)
    return tmax

def otsu_test(ht): #오츠를 하는 코드
    t = []
    for i in range(0,256):
        l = ht[:i]
        r = ht[i:]
        la = np.sum(l)/len(l)
        ra = np.sum(r)/len(r)
        lv = np.sum(pow(l-la,2))
        rv = np.sum(pow(r-ra,2))
        t.append(lv+rv)
    t = np.array(t)
    return np.argmin(t)

def black_img(a,b):   
    imsi2 = np.where(a < b)

    x_List = np.zeros((200,200))
    x_List[imsi2[0],imsi2[1]] = 255

    return x_List

def Lota_size(img):
    m,z,L,v,C = None,None,None,None,None

    Image_zeros = np.zeros((210,210))
    Binary_image_location = np.argwhere(img == 255).transpose()
    m = np.mean(Binary_image_location,1)
    m = m[:, np.newaxis]
    z = Binary_image_location- m
    nn = z.shape[1]
    C = np.dot(z,z.T)/nn
    L,v = lin.eig(C)
    
    Binary_image_location = v.T.dot(Binary_image_location)
    Binary_image_location[0]-=np.min(Binary_image_location[0])
    Binary_image_location[1]-=np.min(Binary_image_location[1])
    Binary_image_location = Binary_image_location.astype(int)
    #print(np.max(Binary_image_location[0]),np.max(Binary_image_location[1]))
    Image_zeros[Binary_image_location[0],Binary_image_location[1]] = 1

    return Image_zeros

def bi(img):
    zs = np.zeros((len(img)+2,len(img[0])+2))
    x = np.arange(1,len(img[0])+1)
    y = np.arange(1,len(img)+1)
    x,y = np.meshgrid(x,y)
    zs[y,x] = img[y-1,x-1]
    x1 = x-1
    x2 = x+1
    y1 = y-1
    y2 = y+1
    ap = y1/(y1+y2)
    bt = y2/(y1+y2)
    q  = x1/(x1+x2)
    p  = x2/(x1+x2)
    P = q*(bt*zs[y2,x1]+ap*zs[y1,x1])+q*(bt*zs[y2,x2]+ap*zs[y1,x2])
    #print(P)
    P = np.where(P == 0 , 0,1)
    return P

def size_up(image):
    zeros_box = np.zeros((len(image),len(image[0])))
    image_location = np.argwhere(image == image).transpose()
    image_location_1 =np.argwhere(image == 1).transpose()
    
    x_size_up = (len(image[0])-1)/np.max(image_location_1[1])
    y_size_up = (len(image)-1)/np.max(image_location_1[0])

    image_location_1 = image_location_1.astype(float)
    image_location_1[1]*=x_size_up
    image_location_1[0]*=y_size_up
    image_location_1 = image_location_1.astype(int)

    zeros_box[image_location_1[0],image_location_1[1]] = 1
    
    return zeros_box

def down_right_sift(image):
    image = bi(image)
    lc = np.argwhere(image[:,:len(image[0])-1] == 1).transpose()
    image[lc[0],lc[1]+1] = 1
    return image

def adaptive_thresh(input_img):

    h, w = input_img.shape

    S = w//8
    s2 = S//2

    print("S:",S)
    print("s2:",s2)
    T = 3.0

    #integral img
    int_img = np.zeros_like(input_img, dtype=np.uint32)
    for col in range(w):
        for row in range(h):
            int_img[row,col] = input_img[0:row,0:col].sum()

    #output img
    out_img = np.zeros_like(input_img)    

    for col in range(w):
        for row in range(h):
            #SxS region
            y0 = int(max(row-s2, 0))
            y1 = int(min(row+s2, h-1))
            x0 = int(max(col-s2, 0))
            x1 = int(min(col+s2, w-1))

            count = (y1-y0)*(x1-x0)

            sum_ = int_img[y1, x1]-int_img[y0, x1]-int_img[y1, x0]+int_img[y0, x0]

            if input_img[row, col]*count < sum_*(100.-T)/100.:
                out_img[row,col] = 0
            else:
                out_img[row,col] = 255

    return out_img

def run():
    fig = plt.figure()
    img = img_read_all()
    subplot_size = np.ceil(np.sqrt(len(img)))
    x_List = np.zeros((200,200))
    print(len(img))
    for i in range(0,len(img)):
        sha = gradeision_remove(mpimg.imread(img[i]))
        his = mser_histo(sha)
        dms = date_mser(his,sha)
        test = mser_histo_see(sha,100)
        ttee = adaptive_thresh(test)
        
        
        #bb = black_img(test,tee)
        #test_lo = Lota_size(dms) # 로테이션
        #bogan = bi(test_lo) # 보간 
        #upsize = size_up(bogan) #사이즈업
        #drs = down_right_sift(upsize) #보간 
        fig.add_subplot(subplot_size, subplot_size, i+1)
        plt.imshow(ttee,'gray')
        plt.axis('off')

        '''
        fig.add_subplot(2, 2, 1)
        plt.imshow(mpimg.imread(img[i]),'gray')
        plt.title('org')
        plt.axis('off')
        fig.add_subplot(2, 2, 2)
        plt.imshow(test,'gray')
        plt.axis('off')
        plt.title('mser')
        fig.add_subplot(2, 2, 4)
        plt.imshow(ttee,'gray')
        plt.axis('off')
        plt.title('smrmaak')
        fig.add_subplot(2, 2, 3)
        plt.imshow(adaptive_thresh(mpimg.imread(img[i])),'gray')
        plt.axis('off')
        plt.title('mser_bradley')
        '''
        print((i+1))

if __name__ == "__main__" : #메인 함수
    run()
    plt.show()
