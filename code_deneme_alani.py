# Python program explaining 
# save() function 
  
import numpy as geek
  
a = geek.arange(5)
  
# a is printed.
print("a is:")
print(a)
  
# the array is saved in the file geekfile.npy 
geek.save('geekfile', a)
  
print("the array is saved in the file geekfile.npy")