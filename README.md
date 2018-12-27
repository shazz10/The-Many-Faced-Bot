# The-Many-Faced-Bot
It is a BOT that can recognise anyone whose Face is present in the database. GOT fans will get it!! ;p

## Introduction
Folks going through deeplearning must have come across the google's fascinating research paper *FaceNet* and it's *Triplet Loss* which is the best right now.
**In this project, I have trained a model on the basis of the paper and used that model to make this BOT.**
### Requirements to run this project
* Tensorflow-gpu
* Keras
* OpenCV
### Instructions to run this project
* Go to terminal and type "python3 many_faced_bot.py"
* Follow the instructions further displayed on the screen(*Press 1 to add face, 2 to detect face, 3 to exit) 
## Milestones
- [x] Make a model and train it on LFW dataset
- [x] Test model for accuracy(more than 99% achieved)
- [x] Save the model and deploy it on CLI application
- [x] Capture and Store Photos in a database and use those to **Recognise**
- [ ] Deploy it on webapp or window-based-app(*for webapp Django is preferred*)
- [ ] For Face-Detection only the **lively face** is detected and not a printed photo or anything
- [ ] Integrate the BOT with Ubuntu-OS and make a Face-Unlock-System
