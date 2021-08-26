from hitable_list import random_scene, three_ball, three_ball_only_lambertian
from hitable import hit_record, sphere
from ray import vec3, ray
import math
from camera import camera
import random
from material import lambertian, material, metal, dielectric
import numpy as np
import cv2
import time
import gc


def color(rays, world, depth=50):

    n = len(rays)

    frame_intensity = vec3(np.ones(n), np.ones(n), np.ones(n))
    frame_rays = rays
    rec = hit_record(n)
    materials = set([x.mat for x in world])
    
    
    
    for i in range(depth):
        #start = time.time()
        
        rec.t.fill(np.inf)
        rec.mat_id.fill(0)

        for s in world:
            s.hit(rays, 0.001, np.inf, rec)
    
        
        for mat in materials:
            mat_indices = np.where(rec.mat_id == id(mat))[0]
            if len(mat_indices) == 0:
                continue

            my_rays = rays[mat_indices]
            my_rec = rec[mat_indices]

            result = mat.scatter(my_rays, my_rec)

            rays[mat_indices] = result.rays
            frame_rays[my_rec.index].direction = result.rays.direction

          
            intensity = result.attenuation.mul(frame_intensity[my_rec.index])
            intensity[np.where(~result.is_scattered)] = vec3(0.,0.,0.)
            
            frame_intensity[my_rec.index] = intensity

            not_scattered_idx = mat_indices[~result.is_scattered]
            rec.t[not_scattered_idx] = np.inf

        scattered_indices = np.where(rec.t != np.inf)[0]
        rays = rays[scattered_indices]
        rec = rec[scattered_indices]

        #end = time.time()
        #print("interval: {0} : {1}ms".format(i, end-start))

        if len(rays) == 0:
            break
        
    unit_direction = frame_rays.direction.unit_vec()
    t = 0.5 * unit_direction.y + 1.0
    img = (vec3(1.0, 1.0, 1.0) * (1.0-t) + vec3(0.5, 0.7, 1.0) * t).mul(frame_intensity)

    return img

def sample_to_image(col,nx, ny, ns):
    x = col.x.reshape([-1,ns]).sum(axis=1)
    y = col.y.reshape([-1,ns]).sum(axis=1)
    z = col.z.reshape([-1,ns]).sum(axis=1)
    x = np.sqrt(x/ns)
    y = np.sqrt(y/ns)
    z = np.sqrt(z/ns)

    im = np.concatenate([x,y,z])
    im = np.reshape(im, (-1,3), order='F')
    im = np.reshape(im, (ny, nx, 3))

    return im

def write_jpg(img, name="sample1.jpg"):
    img = img[:,:,[2,1,0]] * 255.99#cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #cv2.imshow('img', img)
    cv2.imwrite(name, img)

def write_ppm(nx, ny, col):
    img = ["P3\n", str(nx)+" "+str(ny), "\n255\n"]
    for i in range(nx*ny):
        ic = vec3(math.sqrt(col[i].x), math.sqrt(col[i].y), math.sqrt(col[i].z)) * 255.99
        pc = " ".join(map(lambda x: str(int(x)), ic.make_list())) + "\n"
        img.append(pc)
    f = open("sample01.ppm", 'w')
    f.writelines(img)
    f.close()
    

def raytracing():
    nx = 1200
    ny = 800
    ns = 10

    world = random_scene()

    fov = 20
    lookfrom = vec3(13.0, 2.0, 3.0)
    lookat = vec3(0.0, 0.0, 0.0)
    dist_to_focus = 10.0
    aperture = 0.1

    cam = camera(fov, nx, ny, lookfrom, lookat, vec3(0.,1.,0.), aperture, dist_to_focus)

    s = time.time()
    r = cam.get_ray(ns, defocus_blur=True)

    c = color(r, world)
    e = time.time()
    print("time :", e-s)

    #write_ppm(nx,ny,c)
    write_jpg(sample_to_image(c,nx,ny,ns))
    
raytracing()