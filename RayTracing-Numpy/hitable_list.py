from ray import vec3
from hitable import  sphere
from ray import vec3
from material import lambertian, metal, dielectric
import random

def random_scene():
    h_list = [sphere(vec3(0.0, -1000.0, 0.0), 1000.0, lambertian(vec3(0.5, 0.5, 0.5)))]

    for a in range(-11,11):
        for b in range(-11,11):
            choose_mat = random.random()
            center = vec3(a+0.9*random.random(), 0.2, b+0.9*random.random())
            if abs(center - vec3(4, 0.2, 0)) > 0.9:
                if choose_mat < 0.8:
                    m = lambertian(vec3(random.random()*random.random(), 
                        random.random()*random.random(), random.random()*random.random()))
                    h_list.append(sphere(center, 0.2, m))
                elif choose_mat < 0.95:
                    m = metal(vec3(0.5*(1.0+random.random()), 
                        0.5*(1.0+random.random()), 0.5*(1.0+random.random())), 0.5*(random.random()))
                    h_list.append(sphere(center, 0.2, m))
                else:
                    m = dielectric(1.5)
                    h_list.append(sphere(center, 0.2, m))

    m = dielectric(1.5)
    h_list.append(sphere(vec3(0.0,1.0,0.0), 1.0, m))
    m = lambertian(vec3(0.4, 0.2, 0.1))
    h_list.append(sphere(vec3(-4.0, 1.0, 0.0), 1.0, m))
    m = metal(vec3(0.7, 0.6, 0.5), 0.0)
    h_list.append(sphere(vec3(4.0, 1.0, 0.0), 1.0, m))

    return h_list

def three_ball():
    h_list = []

    m = lambertian(vec3(0.8, 0.8, 0.0))
    h_list.append(sphere(vec3(0.0, -100.5, -1.0), 100.0, m))
    
    m = lambertian(vec3(0.1,0.2,0.5))
    h_list.append(sphere(vec3(0.0, 0.0, -1.0), 0.5, m))
    
    m = metal(vec3(0.8, 0.6, 0.2))
    h_list.append(sphere(vec3(1.0, 0.0, -1.0), 0.5, m))
    
    m = dielectric(1.5)
    h_list.append(sphere(vec3(-1.0, 0.0, -1.0), 0.5, m))
    h_list.append(sphere(vec3(-1.0, 0.0, -1.0), -0.45, dielectric(1.5)))
    
    return h_list

def three_ball_only_lambertian():
    h_list = [sphere(vec3(0.0, 0.0, -1.0), 0.5, lambertian(vec3(0.1,0.2,0.5)))]
    h_list.append(sphere(vec3(0, -100.5, -1.0), 100, lambertian(vec3(0.8, 0.8, 0.0))))
    h_list.append(sphere(vec3(1.0, 0.0, -1.0), 0.5, lambertian(vec3(0.8, 0.6, 0.2))))
    h_list.append(sphere(vec3(-1.0, 0.0, -1.0), 0.5, lambertian(vec3(0.2,0.4,0.6))))

    return h_list