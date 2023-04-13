#!/usr/bin/env python
# coding: utf-8

# ### Homework 2. NumPy

# In[1]:


import numpy as np
data = np.loadtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', dtype='object', delimiter=',')
data.shape


# __1. Знайти в датасеті таргет та видалити цю колонку з датасету (видаляти за індексом)__

# In[2]:


data[:10]


# In[3]:


data = np.delete(data, -1, axis=1)


# In[4]:


data[:10]


# __2. Перетворити колонки, що залишились в 2D масив (або впевнитись, що це уже 2D масив)__

# In[5]:


data.ndim


# __3. Порахувати mean, median, standard deviation для 1-ї колонки__
# 

# In[6]:


data = data.astype('float')


# In[7]:


print("Mean value is: ", np.mean(data[:,0]))
print("Median value is: ", np.median(data[:,0]))
print("Standard deviation value is: ", np.std(data[:,0]))


# __4. Вставити 20 значень np.nan на випадкові позиції в масиві 
# (при використанні звичайного рандому можуть накластись позиції, тому знайти рішення, яке гарантує 20 унікальних позицій)__

# In[8]:


rng = np.random.default_rng()
ravel_data = data.ravel()
idx = rng.choice(data.size, size=20, replace=False)
ravel_data[idx] = np.nan


# In[9]:


np.isnan(data).sum()


# __5. Знайти позиції вставлених значень np.nan в 1-й колонці__

# In[10]:


np.where(np.isnan(data[:, 0]))


# __6. Відфільтрувати массив за умовою: значення в 3-й колонці > 1.5 та значения в 1-й колонці < 5.0__

# In[11]:


mask = (data[:, 2] > 1.5) & (data[:, 0] < 5.0)
print(data[mask])


# __7. Замінити всі значення np.nan на 0__

# In[12]:


data[np.isnan(data)] = 0


# In[13]:


np.isnan(data).sum()


# In[14]:


#кількість нулів в масиві (перевірка)
num_zeros = data.size - np.count_nonzero(data)
print(num_zeros)


# __8. Порахувати всі унікальні значення в массиві та вивести їх разом із кількістю__

# In[15]:


uniqs, counts = np.unique(data, return_counts=True)
print("Unique items: ", uniqs)
print("Counts: ", counts)


# __9. Розбити масив по горизонталі на 2 рівні частини (не використовувати абсолютні числа, мають бути два массиви по 4 колонки)__

# In[16]:


data_1, data_2 = np.split(data, 2, axis=0)


# __10. Відсортувати обидва массиви по 1-й колонці: 1-й за збільшенням, 2-й за зменшенням__

# In[17]:


sorted_data_1 = data_1[data_1[:, 0].argsort()]
sorted_data_2 = data_2[data_2[:, 0].argsort()[::-1]]


# __11. Зібрати обидва массиви в одне ціле__

# In[18]:


data = np.concatenate([data_1, data_2], axis=0)


# __12. Знайти найбільш часто повторюване значення в массиві__

# In[19]:


uniqs, counts = np.unique(data, return_counts=True)
uniqs[np.argmax(counts)]


# __13. Написати функцію, яка б множила всі значення в колонці, які менше середнього значения в цій колонці, на 2, і ділила інші значення на 4.__

# In[20]:


def compar_mean(d): 
    return np.select([d < d.mean(), d >= d.mean()], [d * 2, d / 4])


# __14. Застосувати отриману функцію до 3-ї колонки__

# In[21]:


compar_mean(data[:, 2])

