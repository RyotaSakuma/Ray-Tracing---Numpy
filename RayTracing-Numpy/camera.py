from ray import vec3, ray, dot
import math
import random
import numpy as np

def random_in_unit_disk(n):
    theta = np.random.rand(n) * 2.0 * np.pi
    r = np.sqrt(np.random.rand(n))
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return vec3(x,y,0.0)

class camera():
    def __init__(self, fov, nx, ny, lookfrom, lookat, vup, aperture, focus_dist):


        self.nx = nx
        self.ny = ny
        
        self.lens_radius = aperture / 2      
        half_height = math.tan(math.radians(fov/2.0))
        half_width = (nx/ny) * half_height

        self.w = (lookfrom-lookat).unit_vec()
        self.u = (vup.cross(self.w)).unit_vec()
        self.v = self.w.cross(self.u)
        self.nx = nx
        self.ny = ny

        self.lower_left_corner = lookfrom - (self.u * half_width + self.v * half_height + self.w) * focus_dist
        self.horizontal = self.u * (2.0 * half_width * focus_dist)
        self.vertical = self.v * (2.0 * half_height * focus_dist)

        self.origin = lookfrom


    def get_ray(self, ns=1, defocus_blur = True):
        n = self.nx*self.ny*ns
        if defocus_blur:
            rd = random_in_unit_disk(n) * self.lens_radius
            offset = self.u * rd.x + self.v * rd.y
        else:
            offset = vec3(0.,0.,0.)

        x, y = [], []

        for j in range(self.ny-1,-1,-1):
            for i in range(self.nx):
                x += [i]*ns
                y += [j]*ns
        x = np.array(x) + np.random.rand(n)
        y = np.array(y) + np.random.rand(n)
        x = x / self.nx
        y = y / self.ny
                

        origin = self.origin.repeat(n)

        return ray(origin+offset, self.lower_left_corner+self.horizontal*x+self.vertical*y-origin-offset)