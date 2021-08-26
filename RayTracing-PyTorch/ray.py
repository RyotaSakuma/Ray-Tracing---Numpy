import math
import numpy as np

class vec3():
    def __init__(self, x=0.0, y=0.0, z=0.0, np_vec=None):
        if np_vec is not None:
            x, y, z = np_vec
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def where(condition, v1, v2):
        x = np.where(condition, v1.x, v2.x)
        y = np.where(condition, v1.y, v2.y)
        z = np.where(condition, v1.z, v2.z)
        return vec3(x,y,z)

    def __add__(self, v):
        return vec3(self.x+v.x, self.y+v.y, self.z+v.z)

    def __sub__(self, v):
        return vec3(self.x-v.x, self.y-v.y, self.z-v.z)

    def mul(self, v):
        return vec3(self.x*v.x, self.y*v.y, self.z*v.z)

    def __mul__(self, a):
        return vec3(self.x*a, self.y*a, self.z*a)
    

    def div(self, v):
        return vec3(self.x/v.x, self.y/v.y, self.z/v.z)

    
    def __truediv__(self, a):
        return vec3(self.x/a, self.y/a, self.z/a)
    

    def dot(self, v):
        return self.x*v.x + self.y*v.y + self.z*v.z

    def cross(self, v): # self,vは(float,float,float)前提
        return vec3(self.y*v.z-self.z*v.y, self.z*v.x-self.x*v.z, self.x*v.y-self.y*v.x)

    def __abs__(self):
        return np.sqrt(self.dot(self))

    def unit_vec(self):
        mag = abs(self)
        return self / np.where(mag == 0, 1, mag) 

    def repeat(self, n):
        x = np.repeat(self.x, n)
        y = np.repeat(self.y, n)
        z = np.repeat(self.z, n)

        return vec3(x,y,z)

    def make_list(self):
        return [self.x, self.y, self.z]

    def printval(self):
        print((self.x, self.y, self.z))

    def __getitem__(self, idx):
        return vec3(self.x[idx], self.y[idx], self.z[idx])

    def __setitem__(self, idx, other):
        self.x[idx] = other.x
        self.y[idx] = other.y
        self.z[idx] = other.z

    def __len__(self):
        return self.x.size

class ray():
    def __init__(self, a, b):
        self.origin = a
        self.direction = b

    def point_at_parameter(self, t):
        return self.origin + self.direction*t

    def __getitem__(self, idx):
        return ray(self.origin[idx], self.direction[idx])

    def __setitem__(self, idx, other):
        self.origin[idx] = other.origin
        self.direction[idx] = other.direction

    def __len__(self):
        return self.origin.x.size

def dot(a, b):
    return a.x*b.x + a.y*b.y + a.z*b.z