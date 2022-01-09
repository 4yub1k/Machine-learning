# Machine-learning
**Polynomial Regression : (poly_reg.py)**

<img src="https://user-images.githubusercontent.com/45902447/148652849-fb701427-6449-4ae1-9809-1dfb5b7382f9.png" width="300" height="300"><img src="https://user-images.githubusercontent.com/45902447/148652900-b4b70808-8548-45b8-ab94-1c655dc3bf88.png" width="300" height="300">


**Evaluation :**

![eval_poly](https://user-images.githubusercontent.com/45902447/148652871-855f6725-4c21-407d-b7e2-e82c813514da.PNG)

**Non Linear Regression : (non_linear.py)**

<img src="https://user-images.githubusercontent.com/45902447/148679307-0d8d94eb-a007-43b7-acae-56d18876fcdf.png" width="300" height="300"><img src="https://user-images.githubusercontent.com/45902447/148679334-e4f6bdae-a82a-4ef5-8d26-6c6f6aa67f51.png" width="300" height="300"><img src="https://user-images.githubusercontent.com/45902447/148679373-cae919f3-fa7a-4d9f-aa8e-4b86e601dd41.png" width="300" height="300">

**Evaluation :**

![image](https://user-images.githubusercontent.com/45902447/148679446-47d4dee9-824e-4110-9dbf-8dbfc660e4b7.png)

# Equations for Non Linear Regression:
  ```
  import numpy as np
  import matplotlib.pyplot as plt
```
## 1) y = 2x + c
```
x = np.arange(-5.0, 5.0, 0.1)
y = 2*(x) + 3
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
#plt.figure(figsize=(8,6))
plt.plot(x, ydata,  'bp')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()
```
![y=2x+c](https://user-images.githubusercontent.com/45902447/148675603-65129483-12a3-4021-ba69-1fa09f8c1811.PNG)

## 2) ![image](https://user-images.githubusercontent.com/45902447/148676038-789c49be-7cc0-4ee7-b231-1fbdaa5d101c.png)
```
x = np.arange(-5.0, 5.0, 0.1)
##You can adjust the slope and intercept to verify the changes in the graph
y = 1*(x**3) + 1*(x**2) + 1*x + 3
y_noise = 20 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()
```
![image](https://user-images.githubusercontent.com/45902447/148676094-ad3da418-15e4-4eb7-90f7-849bff311373.png)
## 3) Quadratic : ![image](https://user-images.githubusercontent.com/45902447/148676129-fbb72db8-72db-455d-9318-1adaf738fdad.png)
```
x = np.arange(-5.0, 5.0, 0.1)
y = np.power(x,2)
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()
```
![image](https://user-images.githubusercontent.com/45902447/148676161-55b44b2b-564f-4674-9d0a-a1b18c794b1d.png)
## 4) Exponential : ![image](https://user-images.githubusercontent.com/45902447/148676192-0989dbd3-287d-4080-8a6c-6fdccd79392a.png)
```
X = np.arange(-5.0, 5.0, 0.1)
Y= np.exp(X)

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()
```
![image](https://user-images.githubusercontent.com/45902447/148676200-92f13205-b957-4f9b-aead-28ab9ff45b99.png)
## 5) Logarithmic : ![image](https://user-images.githubusercontent.com/45902447/148676224-caaa2fc3-ac16-4220-92fd-1f486c763b63.png)
```
X = np.arange(-5.0, 5.0, 0.1)

Y = np.log(X)

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()
```
![image](https://user-images.githubusercontent.com/45902447/148676245-b8b8a877-571d-4822-8ed4-cfd97ebdd78c.png)
## 6) Sigmoidal/Logistic : ![image](https://user-images.githubusercontent.com/45902447/148676272-e059ae62-98e2-48de-ae14-f17e12f65e6d.png)

```
X = np.arange(-5.0, 5.0, 0.1)
Y = 1-4/(1+np.power(2, X-9))

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()
```
![image](https://user-images.githubusercontent.com/45902447/148676338-313e2efe-7cb2-4fc3-94ca-5fd10599e034.png)
