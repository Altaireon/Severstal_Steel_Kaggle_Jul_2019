from data.lib.include import *
    
def run_length_decode(rle, height=256, width=1600, fill_value=1):
    mask = np.zeros((height,width), np.float32)
    if rle != '':
        mask=mask.reshape(-1)
        r = [int(r) for r in rle.split(' ')]
        r = np.array(r).reshape(-1, 2)
        for start,length in r:
            start = start-1  #???? 0 or 1 index ???
            mask[start:(start + length)] = fill_value
        mask=mask.reshape(width, height).T
    return mask

class FiveBalanceClassSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset

        label = (self.dataset.df['Label'].values)
        label = label.reshape(-1,4)
        label = np.hstack([label.sum(1,keepdims=True)==0,label]).T

        self.neg_index  = np.where(label[0])[0]
        self.pos1_index = np.where(label[1])[0]
        self.pos2_index = np.where(label[2])[0]
        self.pos3_index = np.where(label[3])[0]
        self.pos4_index = np.where(label[4])[0]
        self.num_image = len(self.dataset.df)//4
        self.length = self.num_image*5


    def __iter__(self):
        
        neg  = np.random.choice(self.neg_index,  self.num_image, replace=True)
        pos1 = np.random.choice(self.pos1_index, self.num_image, replace=True)
        pos2 = np.random.choice(self.pos2_index, self.num_image, replace=True)
        pos3 = np.random.choice(self.pos3_index, self.num_image, replace=True)
        pos4 = np.random.choice(self.pos4_index, self.num_image, replace=True)

        l = np.stack([neg,pos1,pos2,pos3,pos4]).T
        l = l.reshape(-1)
        return iter(l)

    def __len__(self):
        return self.length