import numpy as np
import numpy.linalg as LA
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve



def potentialKeypointDetection(pyramidLayer, w=16):
    potentialKeypoint = []

    
    pyramidLayer[:,:,0] = 0
    pyramidLayer[:,:,-1] = 0
    
    for i in range(w//2+1, pyramidLayer.shape[0]-w//2-1):
        for j in range(w//2+1, pyramidLayer.shape[1]-w//2-1):
            for k in range(1, pyramidLayer.shape[2]-1): 
                patch = pyramidLayer[i-1:i+2, j-1:j+2, k-1:k+2]
                #here the central point will have index 13
                
                if np.argmax(patch) == 13 or np.argmin(patch) == 13:
                    potentialKeypoint.append([i, j, k])

    return potentialKeypoint

def localizingKeypoint(pyramidLayer, x, y, s):
    dx = (pyramidLayer[y,x+1,s]-pyramidLayer[y,x-1,s])/2.
    dy = (pyramidLayer[y+1,x,s]-pyramidLayer[y-1,x,s])/2.
    ds = (pyramidLayer[y,x,s+1]-pyramidLayer[y,x,s-1])/2.

    dxx = pyramidLayer[y,x+1,s]-2*pyramidLayer[y,x,s]+pyramidLayer[y,x-1,s]
    dxy = ((pyramidLayer[y+1,x+1,s]-pyramidLayer[y+1,x-1,s]) - (pyramidLayer[y-1,x+1,s]-pyramidLayer[y-1,x-1,s]))/4.
    dxs = ((pyramidLayer[y,x+1,s+1]-pyramidLayer[y,x-1,s+1]) - (pyramidLayer[y,x+1,s-1]-pyramidLayer[y,x-1,s-1]))/4.
    dyy = pyramidLayer[y+1,x,s]-2*pyramidLayer[y,x,s]+pyramidLayer[y-1,x,s]
    dys = ((pyramidLayer[y+1,x,s+1]-pyramidLayer[y-1,x,s+1]) - (pyramidLayer[y+1,x,s-1]-pyramidLayer[y-1,x,s-1]))/4.
    dss = pyramidLayer[y,x,s+1]-2*pyramidLayer[y,x,s]+pyramidLayer[y,x,s-1]

    J = np.array([dx, dy, ds])
    HD = np.array([
        [dxx, dxy, dxs],
        [dxy, dyy, dys],
        [dxs, dys, dss]])
    
    offset = -LA.inv(HD).dot(J) 
    return offset, J, HD[:2,:2], x, y, s

def getPotentialKeypoints(pyramidLayer, R_th, t_c, w):
    potentialKeypoint = potentialKeypointDetection(pyramidLayer, w)
    #print('%d candidate keypoints found' % len(potentialKeypoint))

    keypoints = []

    for i, cand in enumerate(potentialKeypoint):
        y, x, s = cand[0], cand[1], cand[2]
        offset, J, H, x, y, s = localizingKeypoint(pyramidLayer, x, y, s)

        contrast = pyramidLayer[y,x,s] + .5*J.dot(offset)
        if abs(contrast) < t_c: continue

        w, v = LA.eig(H)
        r = w[1]/w[0]
        R = (r+1)**2 / r
        if R > R_th: continue

        kp = np.array([x, y, s]) + offset
        if kp[1] >= pyramidLayer.shape[0] or kp[0] >= pyramidLayer.shape[1]: continue 

        keypoints.append(kp)

    #print('%d keypoints found' % len(keypoints))
    return np.array(keypoints)

def detectKeypoints(DOG_Pyramid, R_th, t_c, w):
    finalKyepoints = []

    for pyramidLayer in DOG_Pyramid:
        finalKyepoints.append(getPotentialKeypoints(pyramidLayer, R_th, t_c, w))

    return finalKyepoints


#using KUM value API
def kum_value():
    for i in range(1,length(kend),2):
        si[ceil(kend[i]/6)][ceil(kend[i+1]/6)]= si[ceil(kend[i]/6)][ceil(kend[i+1]/6)+1]
        no_of_key_points=length(kend)/2;
        key_points_density=no_of_key_points/(ceil(n/6)*ceil(m/6));
        kum_square=0;
        for i in range(1,ceil(m/6)):
            for j in range(1,ceil(n/6)):
                kum_square=kum_square+(key_points_density-si[i][j])*(key_points_density-si[i][j])
            kum=sqrt(kum_square/no_of_key_points)
            thres=thres/2
            if kum <0.3:
                break;

def elli_des():
    for x in range(0,359,45 ):
        magcount=0
        for i in range(-4,4):
            for j in range(-floor(sqrt(16-i*i)),floor(sqrt(16-i*i))):
                
                ch1=-180+x
                ch2=-180+45+x
                if (ch1<0  or  ch2<0):
                	if ((k1+i)>0 and (j1+j)>0 and (k1+i)<m and (j1+j)<n):
                        
                    		if (abs(oric[k1+i][j1+j])<abs(ch1) and abs(oric[k1+i][j1+j])>=abs(ch2)):
                            		if (oric[k1+i][j1+j])>=(ch1) and oric[k1+i][j1+j]<(ch2):
                                		if i<=0 and j<=0:
                                    			if abs(i)>abs(j):
#                                         		y=2 finding x
                                        			if i*i+j*j<=1:
                                            				c[2*8+floor(x/45)+1]=c[2*8+floor(x/45)+1]+mag[k1+i][j1+j]
                                        			elif i*i+j*j <=4:
                                            				c[6*8+floor(x/45)+1]=c[6*8+floor(x/45)+1]+mag[k1+i][j1+j]
                                        			elif i*i+j*j <=9: 
                                            				c[10*8+floor(x/45)+1]=c[10*8+floor(x/45)+1]+mag[k1+i][j1+j]
                                        			elif i*i+j*j <=16: 
                                            				c[14*8+floor(x/45)+1]=c[14*8+floor(x/45)+1]+mag[k1+i][j1+j] 
                                			else:
#                                 			y=1 finding x
                                    				if (i*i+j*j <=1):
                                        				c[1*8+floor(x/45)+1]=c[1*8+floor(x/45)+1]+mag[k1+i][j1+j]
                                    
                                    				elif i*i+j*j <=4:
                                        				c[5*8+floor(x/45)+1]=c[5*8+floor(x/45)+1]+mag[k1+i][j1+j];
                                   
                                    
                                    				elif i*i+j*j <=9: 
                                        				c[9*8+floor(x/45)+1]=c[9*8+floor(x/45)+1]+mag[k1+i][j1+j];
                                    				elif i*i+j*j<=16:
                                        				c[13*8+floor(x/45)+1]=c[13*8+floor(x/45)+1]+mag[k1+i][j1+j];   

                            			if (i>=0 and j<=0):
						        if( abs(i)>=abs(j)):
						       #y=0 finding x
						        	if (i*i+j*j<=1):
									c[0*8+floor(x/45)+1]=c[0*8+floor(x/45)+1]+mag[k1+i][j1+j];
						            
						            	elif i*i+j*j <=4:
						                	c[4*8+floor(x/45)+1]=c[4*8+floor(x/45)+1]+mag[k1+i][j1+j];
						           
						            
						            	elif i*i+j*j <= 9:
						                	c[8*8+floor(x/45)+1]=c[8*8+floor(x/45)+1]+mag[k1+i][j1+j];
						   
						        
						            	elif i*i+j*j <=16:
						            		c[12*8+floor(x/45)+1]=c[12*8+floor(x/45)+1]+mag[k1+i][j1+j];   
						            
						   
						        else:
						        #y= 1
						        	if i*i+j*j<=1:
						               		c[1*8+floor(x/45)+1]=c[1*8+floor(x/45)+1]+mag[k1+i][j1+j];
						            
						           	elif i*i+j*j <=4:
						                	c[5*8+floor(x/45)+1]=c[5*8+floor(x/45)+1]+mag[k1+i][j1+j];
						           
						            
						            	elif i*i+j*j <= 9:
						                	c[9*8+floor(x/45)+1]=c[9*8+floor(x/45)+1]+mag[k1+i][j1+j];
						       
						        
						            	elif i*i+j*j <=16:
						                	c[13*8+floor(x/45)+1]=c[13*8+floor(x/45)+1]+mag[k1+i][j1+j];   
						            
						        
						    
                        
						if (i>=0 and j>=0):
						    	if (abs(i)>abs(j)):
						        
						        	if i*i+j*j<=1:
						              
						                	c[0*8+floor(x/45)+1]=c[0*8+floor(x/45)+1]+mag[k1+i][j1+j];
						            
						            	elif i*i+j*j <=4:
						                	c[4*8+floor(x/45)+1]=c[4*8+floor(x/45)+1]+mag[k1+i][j1+j];
						           
						            
						            	elif i*i+j*j <= 9:
						                	c[8*8+floor(x/45)+1]=c[8*8+floor(x/45)+1]+mag[k1+i][j1+j];
						       
						            
						            	elif i*i+j*j <=16:
						                	c[12*8+floor(x/45)+1]=c[12*8+floor(x/45)+1]+mag[k1+i][j1+j];   

						        else:
						      
						           	if i*i+j*j <=1:
						           	     c[3*8+floor(x/45)+1]=c[3*8+floor(x/45)+1]+mag[k1+i][j1+j];
						            
						            	elif i*i+j*j <=4:
						                	c[7*8+floor(x/45)+1]=c[7*8+floor(x/45)+1]+mag[k1+i][j1+j];
								elif i*i+j*j <= 9:
						                	c[11*8+floor(x/45)+1]=c[11*8+floor(x/45)+1]+mag[k1+i][j1+j];
						       
						            
						            	elif i*i+j*j <=16:
						                	c[15*8+floor(x/45)+1]=c[15*8+floor(x/45)+1]+mag[k1+i][j1+j]; 

						if (i<=0 and j>=0):
						    	if (abs(i)>abs(j)):
						        
						        	if i*i+j*j<=1:
                                            				c[2*8+floor(x/45)+1]=c[2*8+floor(x/45)+1]+mag[k1+i][j1+j]
                                        			elif i*i+j*j <=4:
                                            				c[6*8+floor(x/45)+1]=c[6*8+floor(x/45)+1]+mag[k1+i][j1+j]
                                        			elif i*i+j*j <=9: 
                                            				c[10*8+floor(x/45)+1]=c[10*8+floor(x/45)+1]+mag[k1+i][j1+j]
                                        			elif i*i+j*j <=16: 
                                            				c[14*8+floor(x/45)+1]=c[14*8+floor(x/45)+1]+mag[k1+i][j1+j]   

						        else:
						      
						           	if i*i+j*j <=1:
						           	     c[3*8+floor(x/45)+1]=c[3*8+floor(x/45)+1]+mag[k1+i][j1+j];
						            
						            	elif i*i+j*j <=4:
						                	c[7*8+floor(x/45)+1]=c[7*8+floor(x/45)+1]+mag[k1+i][j1+j];
								elif i*i+j*j <= 9:
						                	c[11*8+floor(x/45)+1]=c[11*8+floor(x/45)+1]+mag[k1+i][j1+j];
						       
						            
						            	elif i*i+j*j <=16:
						                	c[15*8+floor(x/45)+1]=c[15*8+floor(x/45)+1]+mag[k1+i][j1+j];




def getPatchGrads(p):
    r1 = np.zeros_like(p)
    r1[-1] = p[-1]
    r1[:-1] = p[1:]

    r2 = np.zeros_like(p)
    r2[0] = p[0]
    r2[1:] = p[:-1]

    dy = r1-r2

    r1[:,-1] = p[:,-1]
    r1[:,:-1] = p[:,1:]

    r2[:,0] = p[:,0]
    r2[:,1:] = p[:,:-1]

    dx = r1-r2

    return dx, dy

def getHistogramSubregion(m, theta, num_bin, reference_angle, bin_width, subregion_w):
    hist = np.zeros(num_bin, dtype=np.float32)
    c = subregion_w/2 - .5

    for i, (mag, angle) in enumerate(zip(m, theta)):
        angle = (angle-reference_angle) % 360
        binno = quantizeOrientation(angle, num_bin)
        vote = mag

        # binno*bin_width is the start angle of the histogram bin
        # binno*bin_width+bin_width/2 is the center of the histogram bin
        # angle - " is the distance from the angle to the center of the bin 
        hist_interp_weight = 1 - abs(angle - (binno*bin_width + bin_width/2))/(bin_width/2)
        vote *= max(hist_interp_weight, 1e-6)

        gy, gx = np.unravel_index(i, (subregion_w, subregion_w))
        x_interp_weight = max(1 - abs(gx - c)/c, 1e-6)
        y_interp_weight = max(1 - abs(gy - c)/c, 1e-6)
        vote *= x_interp_weight * y_interp_weight

        hist[binno] += vote

    return hist

def localDescriptor(kps, octave, w=16, num_subregion=4, num_bin=8):
    descs = []
    bin_width = 360//num_bin

    for kp in kps:
        cx, cy, s = int(kp[0]), int(kp[1]), int(kp[2])
        s = np.clip(s, 0, octave.shape[2]-1)
        kernel = gaussianFilter(w/6) 
        L = octave[...,s]

        t, l = max(0, cy-w//2), max(0, cx-w//2)
        b, r = min(L.shape[0], cy+w//2+1), min(L.shape[1], cx+w//2+1)
        patch = L[t:b, l:r]

        dx, dy = getPatchGrads(patch)

        if dx.shape[0] < w+1:
            if t == 0: kernel = kernel[kernel.shape[0]-dx.shape[0]:]
            else: kernel = kernel[:dx.shape[0]]
        if dx.shape[1] < w+1:
            if l == 0: kernel = kernel[kernel.shape[1]-dx.shape[1]:]
            else: kernel = kernel[:dx.shape[1]]

        if dy.shape[0] < w+1:
            if t == 0: kernel = kernel[kernel.shape[0]-dy.shape[0]:]
            else: kernel = kernel[:dy.shape[0]]
        if dy.shape[1] < w+1:
            if l == 0: kernel = kernel[kernel.shape[1]-dy.shape[1]:]
            else: kernel = kernel[:dy.shape[1]]

        m, theta = toPolarGrad(dx, dy)
        

        subregion_w = w//num_subregion
        featvec = np.zeros(num_bin * num_subregion**2, dtype=np.float32)

        for i in range(0, subregion_w):
            for j in range(0, subregion_w):
                t, l = i*subregion_w, j*subregion_w
                b, r = min(L.shape[0], (i+1)*subregion_w), min(L.shape[1], (j+1)*subregion_w)

                hist = getHistogramSubregion(m[t:b, l:r].ravel(), 
                                                theta[t:b, l:r].ravel(), 
                                                num_bin, 
                                                kp[3], 
                                                bin_width,
                                                subregion_w)
                featvec[i*subregion_w*num_bin + j*num_bin:i*subregion_w*num_bin + (j+1)*num_bin] = hist.flatten()

        featvec /= max(1e-6, LA.norm(featvec))
        featvec[featvec>0.2] = 0.2
        featvec /= max(1e-6, LA.norm(featvec))
        descs.append(featvec)
 

    return descs


#DOGPYRA

def differenceOfGaussian_Octave(Octave_layer):
    octave = []

    for i in range(1, len(Octave_layer)):
        octave.append(Octave_layer[i] - Octave_layer[i-1])

    return np.concatenate([o[:,:,np.newaxis] for o in octave], axis=2)

def differenceOfGaussian_Pyramid(Gaussian_Pyramid):
    pyr = []

    for Octave_in_pyramid in Gaussian_Pyramid:
        pyr.append(differenceOfGaussian_Octave(Octave_in_pyramid))

    return pyr
    

#filter

def gaussianFilter(sigma):
    size = 2*np.ceil(3*sigma)+1
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2))) / (2*np.pi*sigma**2)
    return g/g.sum()    


#pyramid

def createOcatave(firstLevel, s, sigma):
    octave = [firstLevel]
    #generating different ocataves to form pyramid
    k = 2**(1/s)
    gaussianKernel = gaussianFilter(k * sigma)

    for i in range(s+2):
        nextLevel = convolve(octave[-1], gaussianKernel)
        
        octave.append(nextLevel)

    return octave

def createPyramid(image, octaveNum, s, sigma):
    pyramid = []
    #generating different ocataves pyramid
    for _ in range(octaveNum):
        octave = createOcatave(image, s, sigma)
        pyramid.append(octave)

        image = octave[-3][::2, ::2]

    return pyramid


#orientation

def toPolarGrad(dx, dy):
    m = np.sqrt(dx**2 + dy**2)
    theta = (np.arctan2(dy, dx)+np.pi) * 180/np.pi
    return m, theta

def get_grad(L, x, y):
    dy = L[min(L.shape[0]-1, y+1),x] - L[max(0, y-1),x]
    dx = L[y,min(L.shape[1]-1, x+1)] - L[y,max(0, x-1)]
    return toPolarGrad(dx, dy)

def quantizeOrientation(theta, num_bins):
    bin_width = 360//num_bins
    return int(np.floor(theta)//bin_width)

def fitParabola(hist, binno, bin_width):
    centerval = binno*bin_width + bin_width/2.

    if binno == len(hist)-1: rightval = 360 + bin_width/2.
    else: rightval = (binno+1)*bin_width + bin_width/2.

    if binno == 0: leftval = -bin_width/2.
    else: leftval = (binno-1)*bin_width + bin_width/2.
    
    A = np.array([
        [centerval**2, centerval, 1],
        [rightval**2, rightval, 1],
        [leftval**2, leftval, 1]])
    b = np.array([
        hist[binno],
        hist[(binno+1)%len(hist)], 
        hist[(binno-1)%len(hist)]])

    x = LA.lstsq(A, b, rcond=None)[0]
    if x[0] == 0: x[0] = 1e-6
    return -x[1]/(2*x[0])

def orientationAssignment(kps, octave, num_bins=36):
    new_kps = []
    bin_width = 360//num_bins

    for kp in kps:
        cx, cy, s = int(kp[0]), int(kp[1]), int(kp[2])
        s = np.clip(s, 0, octave.shape[2]-1)

        sigma = kp[2]*1.5
        w = int(2*np.ceil(sigma)+1)
        kernel = gaussianFilter(sigma)

        L = octave[...,s]
        hist = np.zeros(num_bins, dtype=np.float32)

        for oy in range(-w, w+1):
            for ox in range(-w, w+1):
                x, y = cx+ox, cy+oy
                
                if x < 0 or x > octave.shape[1]-1: continue
                elif y < 0 or y > octave.shape[0]-1: continue
                
                m, theta = get_grad(L, x, y)
                weight = kernel[oy+w, ox+w] * m

                bin = quantizeOrientation(theta, num_bins)
                hist[bin] += weight

        max_bin = np.argmax(hist)
        new_kps.append([kp[0], kp[1], kp[2], fitParabola(hist, max_bin, bin_width)])

        max_val = np.max(hist)
        for binno, val in enumerate(hist):
            if binno == max_bin: continue

            if .8 * max_val <= val:
                new_kps.append([kp[0], kp[1], kp[2], fitParabola(hist, binno, bin_width)])

    return np.array(new_kps)



#SIFT

class Improved_SIFT(object):
    def __init__(self, im, s=3, num_octave=4, s0=1.3, sigma=1.6, r_th=10, t_c=0.03, w=16):
        self.im = convolve(rgb2gray(im), gaussianFilter(s0))
        self.s = s
        self.sigma = sigma
        self.num_octave = num_octave
        self.t_c = t_c
        self.R_th = (r_th+1)**2 / r_th
        self.w = w

    def get_features(self):
        gaussian_pyr = createPyramid(self.im, self.num_octave, self.s, self.sigma)
        DOG_Pyramid = differenceOfGaussian_Pyramid(gaussian_pyr)
        kp_pyr = detectKeypoints(DOG_Pyramid, self.R_th, self.t_c, self.w)
        feats = []

        for i, DoG_octave in enumerate(DOG_Pyramid):
            kp_pyr[i] = orientationAssignment(kp_pyr[i], DoG_octave)
            feats.append(localDescriptor(kp_pyr[i], DoG_octave))

        self.kp_pyr = kp_pyr
        self.feats = feats

        return feats
import numpy.linalg as LA
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve



def potentialKeypointDetection(pyramidLayer, w=16):
    potentialKeypoint = []

    
    pyramidLayer[:,:,0] = 0
    pyramidLayer[:,:,-1] = 0
    
    for i in range(w//2+1, pyramidLayer.shape[0]-w//2-1):
        for j in range(w//2+1, pyramidLayer.shape[1]-w//2-1):
            for k in range(1, pyramidLayer.shape[2]-1): 
                patch = pyramidLayer[i-1:i+2, j-1:j+2, k-1:k+2]
                #here the central point will have index 13
                
                if np.argmax(patch) == 13 or np.argmin(patch) == 13:
                    potentialKeypoint.append([i, j, k])

    return potentialKeypoint

def localizingKeypoint(pyramidLayer, x, y, s):
    dx = (pyramidLayer[y,x+1,s]-pyramidLayer[y,x-1,s])/2.
    dy = (pyramidLayer[y+1,x,s]-pyramidLayer[y-1,x,s])/2.
    ds = (pyramidLayer[y,x,s+1]-pyramidLayer[y,x,s-1])/2.

    dxx = pyramidLayer[y,x+1,s]-2*pyramidLayer[y,x,s]+pyramidLayer[y,x-1,s]
    dxy = ((pyramidLayer[y+1,x+1,s]-pyramidLayer[y+1,x-1,s]) - (pyramidLayer[y-1,x+1,s]-pyramidLayer[y-1,x-1,s]))/4.
    dxs = ((pyramidLayer[y,x+1,s+1]-pyramidLayer[y,x-1,s+1]) - (pyramidLayer[y,x+1,s-1]-pyramidLayer[y,x-1,s-1]))/4.
    dyy = pyramidLayer[y+1,x,s]-2*pyramidLayer[y,x,s]+pyramidLayer[y-1,x,s]
    dys = ((pyramidLayer[y+1,x,s+1]-pyramidLayer[y-1,x,s+1]) - (pyramidLayer[y+1,x,s-1]-pyramidLayer[y-1,x,s-1]))/4.
    dss = pyramidLayer[y,x,s+1]-2*pyramidLayer[y,x,s]+pyramidLayer[y,x,s-1]

    J = np.array([dx, dy, ds])
    HD = np.array([
        [dxx, dxy, dxs],
        [dxy, dyy, dys],
        [dxs, dys, dss]])
    
    offset = -LA.inv(HD).dot(J) 
    return offset, J, HD[:2,:2], x, y, s

def getPotentialKeypoints(pyramidLayer, R_th, t_c, w):
    potentialKeypoint = potentialKeypointDetection(pyramidLayer, w)
    #print('%d candidate keypoints found' % len(potentialKeypoint))

    keypoints = []

    for i, cand in enumerate(potentialKeypoint):
        y, x, s = cand[0], cand[1], cand[2]
        offset, J, H, x, y, s = localizingKeypoint(pyramidLayer, x, y, s)

        contrast = pyramidLayer[y,x,s] + .5*J.dot(offset)
        if abs(contrast) < t_c: continue

        w, v = LA.eig(H)
        r = w[1]/w[0]
        R = (r+1)**2 / r
        if R > R_th: continue

        kp = np.array([x, y, s]) + offset
        if kp[1] >= pyramidLayer.shape[0] or kp[0] >= pyramidLayer.shape[1]: continue 

        keypoints.append(kp)

    #print('%d keypoints found' % len(keypoints))
    return np.array(keypoints)

def detectKeypoints(DOG_Pyramid, R_th, t_c, w):
    finalKyepoints = []

    for pyramidLayer in DOG_Pyramid:
        finalKyepoints.append(getPotentialKeypoints(pyramidLayer, R_th, t_c, w))

    return finalKyepoints


#using KUM value API
def kum_value():
    for i in range(1,length(kend),2):
        si[ceil(kend[i]/6)][ceil(kend[i+1]/6)]= si[ceil(kend[i]/6)][ceil(kend[i+1]/6)+1]
        no_of_key_points=length(kend)/2;
        key_points_density=no_of_key_points/(ceil(n/6)*ceil(m/6));
        kum_square=0;
        for i in range(1,ceil(m/6)):
            for j in range(1,ceil(n/6)):
                kum_square=kum_square+(key_points_density-si[i][j])*(key_points_density-si[i][j])
            kum=sqrt(kum_square/no_of_key_points)
            thres=thres/2
            if kum <0.3:
                break;

def elli_des():
    for x in range(0,359,45 ):
        magcount=0
        for i in range(-4,4):
            for j in range(-floor(sqrt(16-i*i)),floor(sqrt(16-i*i))):
                
                ch1=-180+x
                ch2=-180+45+x
                if (ch1<0  or  ch2<0):
                	if ((k1+i)>0 and (j1+j)>0 and (k1+i)<m and (j1+j)<n):
                        
                    		if (abs(oric[k1+i][j1+j])<abs(ch1) and abs(oric[k1+i][j1+j])>=abs(ch2)):
                            		if (oric[k1+i][j1+j])>=(ch1) and oric[k1+i][j1+j]<(ch2):
                                		if i<=0 and j<=0:
                                    			if abs(i)>abs(j):
#                                         		y=2 finding x
                                        			if i*i+j*j<=1:
                                            				c[2*8+floor(x/45)+1]=c[2*8+floor(x/45)+1]+mag[k1+i][j1+j]
                                        			elif i*i+j*j <=4:
                                            				c[6*8+floor(x/45)+1]=c[6*8+floor(x/45)+1]+mag[k1+i][j1+j]
                                        			elif i*i+j*j <=9: 
                                            				c[10*8+floor(x/45)+1]=c[10*8+floor(x/45)+1]+mag[k1+i][j1+j]
                                        			elif i*i+j*j <=16: 
                                            				c[14*8+floor(x/45)+1]=c[14*8+floor(x/45)+1]+mag[k1+i][j1+j] 
                                			else:
#                                 			y=1 finding x
                                    				if (i*i+j*j <=1):
                                        				c[1*8+floor(x/45)+1]=c[1*8+floor(x/45)+1]+mag[k1+i][j1+j]
                                    
                                    				elif i*i+j*j <=4:
                                        				c[5*8+floor(x/45)+1]=c[5*8+floor(x/45)+1]+mag[k1+i][j1+j];
                                   
                                    
                                    				elif i*i+j*j <=9: 
                                        				c[9*8+floor(x/45)+1]=c[9*8+floor(x/45)+1]+mag[k1+i][j1+j];
                                    				elif i*i+j*j<=16:
                                        				c[13*8+floor(x/45)+1]=c[13*8+floor(x/45)+1]+mag[k1+i][j1+j];   

                            			if (i>=0 and j<=0):
						        if( abs(i)>=abs(j)):
						       #y=0 finding x
						        	if (i*i+j*j<=1):
									c[0*8+floor(x/45)+1]=c[0*8+floor(x/45)+1]+mag[k1+i][j1+j];
						            
						            	elif i*i+j*j <=4:
						                	c[4*8+floor(x/45)+1]=c[4*8+floor(x/45)+1]+mag[k1+i][j1+j];
						           
						            
						            	elif i*i+j*j <= 9:
						                	c[8*8+floor(x/45)+1]=c[8*8+floor(x/45)+1]+mag[k1+i][j1+j];
						   
						        
						            	elif i*i+j*j <=16:
						            		c[12*8+floor(x/45)+1]=c[12*8+floor(x/45)+1]+mag[k1+i][j1+j];   
						            
						   
						        else:
						        #y= 1
						        	if i*i+j*j<=1:
						               		c[1*8+floor(x/45)+1]=c[1*8+floor(x/45)+1]+mag[k1+i][j1+j];
						            
						           	elif i*i+j*j <=4:
						                	c[5*8+floor(x/45)+1]=c[5*8+floor(x/45)+1]+mag[k1+i][j1+j];
						           
						            
						            	elif i*i+j*j <= 9:
						                	c[9*8+floor(x/45)+1]=c[9*8+floor(x/45)+1]+mag[k1+i][j1+j];
						       
						        
						            	elif i*i+j*j <=16:
						                	c[13*8+floor(x/45)+1]=c[13*8+floor(x/45)+1]+mag[k1+i][j1+j];   
						            
						        
						    
                        
						if (i>=0 and j>=0):
						    	if (abs(i)>abs(j)):
						        
						        	if i*i+j*j<=1:
						              
						                	c[0*8+floor(x/45)+1]=c[0*8+floor(x/45)+1]+mag[k1+i][j1+j];
						            
						            	elif i*i+j*j <=4:
						                	c[4*8+floor(x/45)+1]=c[4*8+floor(x/45)+1]+mag[k1+i][j1+j];
						           
						            
						            	elif i*i+j*j <= 9:
						                	c[8*8+floor(x/45)+1]=c[8*8+floor(x/45)+1]+mag[k1+i][j1+j];
						       
						            
						            	elif i*i+j*j <=16:
						                	c[12*8+floor(x/45)+1]=c[12*8+floor(x/45)+1]+mag[k1+i][j1+j];   

						        else:
						      
						           	if i*i+j*j <=1:
						           	     c[3*8+floor(x/45)+1]=c[3*8+floor(x/45)+1]+mag[k1+i][j1+j];
						            
						            	elif i*i+j*j <=4:
						                	c[7*8+floor(x/45)+1]=c[7*8+floor(x/45)+1]+mag[k1+i][j1+j];
								elif i*i+j*j <= 9:
						                	c[11*8+floor(x/45)+1]=c[11*8+floor(x/45)+1]+mag[k1+i][j1+j];
						       
						            
						            	elif i*i+j*j <=16:
						                	c[15*8+floor(x/45)+1]=c[15*8+floor(x/45)+1]+mag[k1+i][j1+j]; 

						if (i<=0 and j>=0):
						    	if (abs(i)>abs(j)):
						        
						        	if i*i+j*j<=1:
                                            				c[2*8+floor(x/45)+1]=c[2*8+floor(x/45)+1]+mag[k1+i][j1+j]
                                        			elif i*i+j*j <=4:
                                            				c[6*8+floor(x/45)+1]=c[6*8+floor(x/45)+1]+mag[k1+i][j1+j]
                                        			elif i*i+j*j <=9: 
                                            				c[10*8+floor(x/45)+1]=c[10*8+floor(x/45)+1]+mag[k1+i][j1+j]
                                        			elif i*i+j*j <=16: 
                                            				c[14*8+floor(x/45)+1]=c[14*8+floor(x/45)+1]+mag[k1+i][j1+j]   

						        else:
						      
						           	if i*i+j*j <=1:
						           	     c[3*8+floor(x/45)+1]=c[3*8+floor(x/45)+1]+mag[k1+i][j1+j];
						            
						            	elif i*i+j*j <=4:
						                	c[7*8+floor(x/45)+1]=c[7*8+floor(x/45)+1]+mag[k1+i][j1+j];
								elif i*i+j*j <= 9:
						                	c[11*8+floor(x/45)+1]=c[11*8+floor(x/45)+1]+mag[k1+i][j1+j];
						       
						            
						            	elif i*i+j*j <=16:
						                	c[15*8+floor(x/45)+1]=c[15*8+floor(x/45)+1]+mag[k1+i][j1+j];


#descriptor

def getPatchGrads(p):
    r1 = np.zeros_like(p)
    r1[-1] = p[-1]
    r1[:-1] = p[1:]

    r2 = np.zeros_like(p)
    r2[0] = p[0]
    r2[1:] = p[:-1]

    dy = r1-r2

    r1[:,-1] = p[:,-1]
    r1[:,:-1] = p[:,1:]

    r2[:,0] = p[:,0]
    r2[:,1:] = p[:,:-1]

    dx = r1-r2

    return dx, dy

def getHistogramSubregion(m, theta, num_bin, reference_angle, bin_width, subregion_w):
    hist = np.zeros(num_bin, dtype=np.float32)
    c = subregion_w/2 - .5

    for i, (mag, angle) in enumerate(zip(m, theta)):
        angle = (angle-reference_angle) % 360
        binno = quantizeOrientation(angle, num_bin)
        vote = mag

        # binno*bin_width is the start angle of the histogram bin
        # binno*bin_width+bin_width/2 is the center of the histogram bin
        # angle - " is the distance from the angle to the center of the bin 
        hist_interp_weight = 1 - abs(angle - (binno*bin_width + bin_width/2))/(bin_width/2)
        vote *= max(hist_interp_weight, 1e-6)

        gy, gx = np.unravel_index(i, (subregion_w, subregion_w))
        x_interp_weight = max(1 - abs(gx - c)/c, 1e-6)
        y_interp_weight = max(1 - abs(gy - c)/c, 1e-6)
        vote *= x_interp_weight * y_interp_weight

        hist[binno] += vote

    return hist

def localDescriptor(kps, octave, w=16, num_subregion=4, num_bin=8):
    descs = []
    bin_width = 360//num_bin

    for kp in kps:
        cx, cy, s = int(kp[0]), int(kp[1]), int(kp[2])
        s = np.clip(s, 0, octave.shape[2]-1)
        kernel = gaussianFilter(w/6) 
        L = octave[...,s]

        t, l = max(0, cy-w//2), max(0, cx-w//2)
        b, r = min(L.shape[0], cy+w//2+1), min(L.shape[1], cx+w//2+1)
        patch = L[t:b, l:r]

        dx, dy = getPatchGrads(patch)

        if dx.shape[0] < w+1:
            if t == 0: kernel = kernel[kernel.shape[0]-dx.shape[0]:]
            else: kernel = kernel[:dx.shape[0]]
        if dx.shape[1] < w+1:
            if l == 0: kernel = kernel[kernel.shape[1]-dx.shape[1]:]
            else: kernel = kernel[:dx.shape[1]]

        if dy.shape[0] < w+1:
            if t == 0: kernel = kernel[kernel.shape[0]-dy.shape[0]:]
            else: kernel = kernel[:dy.shape[0]]
        if dy.shape[1] < w+1:
            if l == 0: kernel = kernel[kernel.shape[1]-dy.shape[1]:]
            else: kernel = kernel[:dy.shape[1]]

        m, theta = toPolarGrad(dx, dy)
        

        subregion_w = w//num_subregion
        featvec = np.zeros(num_bin * num_subregion**2, dtype=np.float32)

        for i in range(0, subregion_w):
            for j in range(0, subregion_w):
                t, l = i*subregion_w, j*subregion_w
                b, r = min(L.shape[0], (i+1)*subregion_w), min(L.shape[1], (j+1)*subregion_w)

                hist = getHistogramSubregion(m[t:b, l:r].ravel(), 
                                                theta[t:b, l:r].ravel(), 
                                                num_bin, 
                                                kp[3], 
                                                bin_width,
                                                subregion_w)
                featvec[i*subregion_w*num_bin + j*num_bin:i*subregion_w*num_bin + (j+1)*num_bin] = hist.flatten()

        featvec /= max(1e-6, LA.norm(featvec))
        featvec[featvec>0.2] = 0.2
        featvec /= max(1e-6, LA.norm(featvec))
        descs.append(featvec)
 

    return descs


#DOGPYRA

def differenceOfGaussian_Octave(Octave_layer):
    octave = []

    for i in range(1, len(Octave_layer)):
        octave.append(Octave_layer[i] - Octave_layer[i-1])

    return np.concatenate([o[:,:,np.newaxis] for o in octave], axis=2)

def differenceOfGaussian_Pyramid(Gaussian_Pyramid):
    pyr = []

    for Octave_in_pyramid in Gaussian_Pyramid:
        pyr.append(differenceOfGaussian_Octave(Octave_in_pyramid))

    return pyr
    

#filter

def gaussianFilter(sigma):
    size = 2*np.ceil(3*sigma)+1
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2))) / (2*np.pi*sigma**2)
    return g/g.sum()    


#pyramid

def createOcatave(firstLevel, s, sigma):
    octave = [firstLevel]
    #generating different ocataves to form pyramid
    k = 2**(1/s)
    gaussianKernel = gaussianFilter(k * sigma)

    for i in range(s+2):
        nextLevel = convolve(octave[-1], gaussianKernel)
        
        octave.append(nextLevel)

    return octave

def createPyramid(image, octaveNum, s, sigma):
    pyramid = []
    #generating different ocataves pyramid
    for _ in range(octaveNum):
        octave = createOcatave(image, s, sigma)
        pyramid.append(octave)

        image = octave[-3][::2, ::2]

    return pyramid


#orientation

def toPolarGrad(dx, dy):
    m = np.sqrt(dx**2 + dy**2)
    theta = (np.arctan2(dy, dx)+np.pi) * 180/np.pi
    return m, theta

def get_grad(L, x, y):
    dy = L[min(L.shape[0]-1, y+1),x] - L[max(0, y-1),x]
    dx = L[y,min(L.shape[1]-1, x+1)] - L[y,max(0, x-1)]
    return toPolarGrad(dx, dy)

def quantizeOrientation(theta, num_bins):
    bin_width = 360//num_bins
    return int(np.floor(theta)//bin_width)

def fitParabola(hist, binno, bin_width):
    centerval = binno*bin_width + bin_width/2.

    if binno == len(hist)-1: rightval = 360 + bin_width/2.
    else: rightval = (binno+1)*bin_width + bin_width/2.

    if binno == 0: leftval = -bin_width/2.
    else: leftval = (binno-1)*bin_width + bin_width/2.
    
    A = np.array([
        [centerval**2, centerval, 1],
        [rightval**2, rightval, 1],
        [leftval**2, leftval, 1]])
    b = np.array([
        hist[binno],
        hist[(binno+1)%len(hist)], 
        hist[(binno-1)%len(hist)]])

    x = LA.lstsq(A, b, rcond=None)[0]
    if x[0] == 0: x[0] = 1e-6
    return -x[1]/(2*x[0])

def orientationAssignment(kps, octave, num_bins=36):
    new_kps = []
    bin_width = 360//num_bins

    for kp in kps:
        cx, cy, s = int(kp[0]), int(kp[1]), int(kp[2])
        s = np.clip(s, 0, octave.shape[2]-1)

        sigma = kp[2]*1.5
        w = int(2*np.ceil(sigma)+1)
        kernel = gaussianFilter(sigma)

        L = octave[...,s]
        hist = np.zeros(num_bins, dtype=np.float32)

        for oy in range(-w, w+1):
            for ox in range(-w, w+1):
                x, y = cx+ox, cy+oy
                
                if x < 0 or x > octave.shape[1]-1: continue
                elif y < 0 or y > octave.shape[0]-1: continue
                
                m, theta = get_grad(L, x, y)
                weight = kernel[oy+w, ox+w] * m

                bin = quantizeOrientation(theta, num_bins)
                hist[bin] += weight

        max_bin = np.argmax(hist)
        new_kps.append([kp[0], kp[1], kp[2], fitParabola(hist, max_bin, bin_width)])

        max_val = np.max(hist)
        for binno, val in enumerate(hist):
            if binno == max_bin: continue

            if .8 * max_val <= val:
                new_kps.append([kp[0], kp[1], kp[2], fitParabola(hist, binno, bin_width)])

    return np.array(new_kps)



#SIFT

class Improved_SIFT(object):
    def __init__(self, im, s=3, num_octave=4, s0=1.3, sigma=1.6, r_th=10, t_c=0.03, w=16):
        self.im = convolve(rgb2gray(im), gaussianFilter(s0))
        self.s = s
        self.sigma = sigma
        self.num_octave = num_octave
        self.t_c = t_c
        self.R_th = (r_th+1)**2 / r_th
        self.w = w

    def get_features(self):
        gaussian_pyr = createPyramid(self.im, self.num_octave, self.s, self.sigma)
        DOG_Pyramid = differenceOfGaussian_Pyramid(gaussian_pyr)
        kp_pyr = detectKeypoints(DOG_Pyramid, self.R_th, self.t_c, self.w)
        feats = []

        for i, DoG_octave in enumerate(DOG_Pyramid):
            kp_pyr[i] = orientationAssignment(kp_pyr[i], DoG_octave)
            feats.append(localDescriptor(kp_pyr[i], DoG_octave))

        self.kp_pyr = kp_pyr
        self.feats = feats

        return feats
