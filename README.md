# 100 Days Of ML/DL Code

## The vlog accompanying this challenge can be found [here](https://www.youtube.com/watch?v=C-XdamS0-zw&list=PLnXc6tX6p6dbBVtb-B_OgAodhQoBBOWrx&index=1).

### Day 1: July 18, 2018

**Today's Progress**: Watched [Siraj Raval's music generation video](https://www.youtube.com/watch?v=4DMm5Lhey1U&t=307s), learning how to create music using deep learning. I learned more about LSTM's, but couldn't get the code running on my end.

**Thoughts:** The video was difficult to understand and I couldn't get all that I wanted out of it. I did, however, understand how LSTM layers work and know how I would use them.

**Link to work:** No work produced today. I hope to get into creating a music generator.

### Day 2: July 19, 2018

**Today's Progress**: Watched [Jordan Boyd-Graber's video on intro to ML](https://www.youtube.com/watch?v=7DjSOLW-ozc&list=PLegWUnz91WfsELyRcZ7d1GwAVifDaZmgo&index=1), learning about the basics of machine learning and the K-Nearest-Neighbors algorithm.
I also created the beginnings of a program that uses ML algorithms to predict whether a person will donate blood again or not. I am completing [this challenge](https://www.drivendata.org/competitions/2/warm-up-predict-blood-donations/page/5/) for my first project.

**Thoughts:** One thing in the video that was interesting was that Jordan said that everything is data, including our genome and the brain itself.
I got to learn about pandas, numpy, and reading in data from a csv and parsing it into the data needed to feed it to a ML algorithm. I have a decent understanding
of this process, but doing it myself and looking up the documentation will help me in the future to understand how the process works of getting data to fit and train models and then
predictingh the labels based off of that input.

**Link to work:** [Here is the file I created](https://github.com/MrJustPeachy/ML100/blob/master/Blood%20Estimator/main.py)

# Day 3: July 20, 2018

**Today's Progress**: Continuing with the blood prediction project from yesterday, I made a file that reads in the data to a Logistic Regreession algorithm and it outputs the probabilities to a CSV file. I submitted the file and scored #1430 out of #4650. Not bad for such a basic algorithm!

**Thoughts:** I feel really confident with my skills after placing that high. I looked up all of the documentation and it gave me the feeling of control and understanding of what I was doing. Obviously this project was super simple, but I now know with certainty that I have the basics down. It really is easy to get a ML algorithm up and going. I want to extend this program to use deep learning techniques and get more practice with that, because I feel much less comfortable with DL libraries, functions, and architectures.

**Link to work:** [Here's the current working file](https://github.com/MrJustPeachy/ML100/blob/master/Blood%20Estimator/main_clunky.py)

# Day 4: July 21, 2018

**Today's Progress**: I narrowed down the program to use only one algorithm - logistic regression. I then implemented GridSearchCV to select the best parameters for the job.

**Thoughts:** I enjoyed refining my solution and using GridSearchCV. I'm going to now move onto using deep learning and switch to a new project.

**Link to work:** [Here's the current working file](https://github.com/MrJustPeachy/ML100/blob/master/Blood%20Estimator/main_clunky.py)

# Day 5: July 22, 2018

**Today's Progress**: I started on a [new kaggle project](https://www.kaggle.com/c/digit-recognizer), and am [following this tutorial that creates a deep neural net using keras](https://www.kaggle.com/poonaml/deep-neural-network-keras-way). I got about halfway through the tutorial today.

**Thoughts:** I learned a lot through doing this kernel. It taught me a lot about keras and how to use it. I thought it was pretty easy to begin making a neural net (compared to tensorflow). Walking through the tutorial and trying to explain everything on the vlog forced me to think through everthing and taught me a lot. I feel much more confident in creating a neural net than I did before I started.

**Link to work:** [Here's the current file](https://github.com/MrJustPeachy/ML100/blob/master/Digit%20Recognizer/Deep%20Learning%20with%20Keras.ipynb)

# Day 6: July 23, 2018

**Today's Progress**: I started making a jupyter notebook with tensorflow. I wasn't able to finish it, but I got pretty far. I looked up all the documentation to understand what I needed.

**Thoughts:** I felt like it was easy to learn how to use tensorflow since it used Keras to create the neural net.

**Link to work:** [Here's the current file](https://github.com/MrJustPeachy/100-days-of-ML-DL-code/blob/master/Digit%20Recognizer/Deep%20Learning%20with%20Tensorflow.ipynb)

# Day 7: July 24, 2018

**Today's Progress**: I tried implementing the final part of a tensorflow solution, but got very lost in the settings related to training, evaluting, and deploying to a production environment. I decided to call it quits after trying to find a solution to get my tensorflow model up and running.

**Thoughts:** I was surprised to find just how difficult it was to get a tensorflow model running. For the time being I don't want to mess with it, because it's outside of my scope. I'm going to stick with Keras. Overall, it was good learning experience and I have a greater appreciation for Keras now.

**Link to work:** [Here's the current file](https://github.com/MrJustPeachy/100-days-of-ML-DL-code/blob/master/Digit%20Recognizer/Deep%20Learning%20with%20Tensorflow.ipynb)

# Day 8: July 26, 2018

**Today's Progress**: I started on the [humpback whale identitification challenge](https://www.kaggle.com/c/whale-categorization-playground). I quickly realized that I was way over my head and am starting to go through different kernels to learn and work on the challenge. 

**Thoughts:** I quickly became confused and didn't have much of a clue as to what I was doing. I still kept on working through it and started by doing the things that I knew had to do (with regards to data/feature analysis. I'd like to do the problem myself by myself but I don't have those skills currently, so the kernel is needed. 

**Link to work:** 

# Day 9: July 30, 2018

**Today's Progress**: Started reading a couple different books. One is Deep Learning by Patterson and Gibson, and the other is Artificial Intelligence for dummies. I really like both for their own reasons.

**Thoughts:** I enjoyed the dummies book discussion on where data comes from and how the AI winters happened. It dived into some of the history of AI that I hadn't learned about before. They also questioned whether hype was justified for deep learning, and I would say it is. The technology really can do a lot of things and even if computers stopped improving, there would be new deep learning techniques for specific types of applications. The deep learning book just reviewed linear algebra and started talking about how machine learning works. I feel like I know this stuff by heart, but I feel like more repetition can only help me improve.

# Day 10: July 31, 2018

**Today's Progress**: Continued reading the deep learning and AI for dummies book.

**Thoughts:** The dummies book continued talking about data and where it comes from. It also discussed the theory of what to do when there is missing data - something which I haven't put a lot of thought into. The deep learning book went on with machine learning and I got a much needed refresher about cost functions because that is one of my weak points with ML.

### I took a break but will be returning to this journal soon enough - my current goal is going to be to [recreate this web app](https://github.com/reiinakano/fast-style-transfer-deeplearnjs) using Node.JS and tensorflow.JS. This repo used deeplearn.js which is now deprecated, so I'm going to be changing it to work with tensorflow.JS!
