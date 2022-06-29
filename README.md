# MeatClassifier
Kaggle dataset: https://www.kaggle.com/datasets/crowww/meat-quality-assessment-based-on-deep-learning

The picture files uploaded in the repo are only a few of the pictures on the Kaggle website.

This was a personal project that I did using the concepts I had learned from watching YouTube videos and doing some googling on my own, regarding
tensorflow, Streamlit, and how to make a convolutional neural network. Gabriel Atkin's "Data Every Day" YouTube series were excellent videos that
I referred back to time and time again. It appears that the images in the dataset were the same sample of meat taken over a series of time as it
began to expire. The dataset authors had 2 photo directories, one being the sample of meat while it was still fresh, and the other being the sample 
of meat once it had expired. 

The goal was to build a small web app in Streamlit where a user could upload one of the images from the dataset, and the model would accurrately
classify whether the meat was spoiled or fresh in the Streamlit web app. The model was an example of transfer learning where I used the mobilenetv2
model and added a couple of Dense layers at the end before passing the image through a sigmoid function for binary classification. The overall
accuracy in my testing was about 98-99% correct for the test set. I had split up the fresh and spoiled images into a dataframe, shuffled the data,
and then used sklearn for splitting into train and test sets. Since both sets were passed through an image generator, the training set also included
a validation set when training. The confusion matrix and classification report are included in the repository, and both are shown when running
the Streamlit app. It should be noted that the images that were used for training and testing were taken in a specific way, so while you could
theoretically upload any image of meat, it wouldn't correctly classify it because the image was not photographed in the way it was when training
and testing. So, the model would get confused. Nevertheless, this project was a good learning experience, and it could be used in a similar fashion 
for a company who takes automated pictures of the samples and is looking to detect expiration.

I was concerned that the timestamped black bar in the top left corner of the supplied data images was unduly influencing the model's prediction.
To test for this, I cropped the photos and retrained the model and tested, yet I still achieved around a 98.6% accuracy although the confusion matrix
showed 3 false negatives and 2 false positives as opposed to the original 1 false negative and 4 false positives. I realized that the timestamp would
be a problem after I had deployed my app. Nonetheless, it appears from my Kaggle testing that the accuracy is not significantly affected so I
left my Github files and results alone and didn't update them. I also didn't redeploy through Heroku.


Deployment was done with Heroku and these two links were helpful: https://gilberttanner.com/blog/deploying-your-streamlit-dashboard-with-heroku/ https://www.youtube.com/watch?v=nJHrSvYxzjE

The citation for the dataset and the kaggle link are also included in the web app as well as my Kaggle profile and GitHub repository for this app.

Citation:
@inproceedings{ulucan2019meat,
title={Meat quality assessment based on deep learning},
author={Ulucan, Oguzhan and Karakaya, Diclehan and Turkan, Mehmet},
booktitle={2019 Innovations in Intelligent Systems and Applications Conference (ASYU)},
pages={1--5},
year={2019},
organization={IEEE}
}

O.Ulucan , D.Karakaya and M.Turkan.(2019) Meat quality assessment based on deep learning.
In Conf. Innovations Intell. Syst. Appli. (ASYU)
