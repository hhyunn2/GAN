class DataGenerator(data.Dataset):
    def __init__(self, img_path, csv, transform = None):
        self.img_path = img_path
        self.csv = csv
        self.transform = transform
        

    def __len__(self):
        return(len(self.csv))

    def __getitem__(self, idx):
        img_id = self.csv.iloc[idx]['img_id']

        # make one-hot vector
        one_hot = [0, 0, 0, 0, 0]

        for i in range(len(self.csv.iloc[idx]) - 1):
            if self.csv.iloc[idx][i+1] == 1:
                one_hot[i] = 1
            else:
                continue

        one_hot = np.array(one_hot)
        one_hot = torch.FloatTensor(one_hot)
        
        # image 
        img = get_img(img_id, self.img_path)
        img = Image.fromarray(img, 'RGB')
        if self.transform == 'transform':
            img = transform(img)

        return img, one_hot
