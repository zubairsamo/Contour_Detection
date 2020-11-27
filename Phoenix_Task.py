#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
import numpy as np
import matplotlib.pyplot as plt


# In[7]:


# Let's load a simple image with Bullet Holes  
image = cv2.imread('C:/Users/zubi_/Desktop/Detect/Image.jpg') 


# In[8]:


plt.imshow(image)


# In[9]:


# Grayscale 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[10]:


plt.imshow(gray)


# In[11]:


# Find Canny edges 
edged = cv2.Canny(gray, 30, 200)
plt.imshow(edged)


# In[12]:


# Finding Contours 
# Use a copy of the image e.g. edged.copy() 
# since findContours alters the image 
contours, hierarchy = cv2.findContours(edged, 
	cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

plt.imshow(edged) 


# In[14]:


# Draw all contours 
# signifies drawing all contours 
Final_img=cv2.drawContours(image, contours, -1, (255, 0, 0), 3) 
plt.imshow(Final_img)


# #  So more approaches to pre process images are down below 
# # First we have to pre process data with atleasr more than 3K or 5k images then train data with resnet50 model through Mark RCNN in shaa allah then we implement some type through cameras i will do my best to full fill your requirment

# In[15]:


# Another Approach 
image = cv2.imread('C:/Users/zubi_/Desktop/Detect/Image.jpg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(1, figsize=(12,8))
plt.imshow(image)


# In[17]:


g = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
edge = cv2.Canny(g, 60, 180)
fig, ax = plt.subplots(1, figsize=(12,8))
plt.imshow(edge, cmap='Greys')


# In[18]:


contours = cv2.findContours(edge, 
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_NONE)
cv2.drawContours(image, contours[0], -1, (0,0,255), thickness = 2)
fig, ax = plt.subplots(1, figsize=(12,8))
plt.imshow(image)


# In[ ]:





# In[20]:


image = cv2.imread('C:/Users/zubi_/Desktop/Detect/Image.jpg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
r, t = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
contours, h = cv2.findContours(t, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
cv2.drawContours(image, contours, -1, (0,0,255), thickness = 5)
fig, ax = plt.subplots(1, figsize=(12,8))
plt.imshow(image)


# In[22]:


# Here i Find Edges Using Convex HUl 
image = cv2.imread('C:/Users/zubi_/Desktop/Detect/Image.jpg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
g = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
edge = cv2.Canny(g, 140, 210)
contours, hierarchy = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for c in contours:
    hull = cv2.convexHull(c)
    cv2.drawContours(image, [hull], 0, (255, 0, 0), 2)
fig, ax = plt.subplots(1, figsize=(12,8))
plt.imshow(image)


# In[ ]:




