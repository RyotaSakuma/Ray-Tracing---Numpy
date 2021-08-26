from hitable_list import hitable_list, random_scene, three_ball
from hitable import hit_record, sphere
from ray import vec3, ray
import math
from camera import camera
import random
from material import lambertian, metal, dielectric

INF = 10.0**15

def color(r, world, depth):
    rec = world.hit(r, 0.001, INF)
    if rec is not None:
        scattered, attenuation = rec.mat.scatter(r, rec)
        if depth < 50 and scattered is not None and attenuation is not None:
            return attenuation.mul(color(scattered, world, depth+1))
        else:
            return vec3(0.0, 0.0, 0.0)

    else:
        unit_direction = r.direction().unit_vec()
        t = 0.5 * (unit_direction.y + 1.0)
        return vec3(1.0, 1.0, 1.0) * (1.0 - t) + vec3(0.5, 0.7, 1.0) * t

def raytracing():
    
    nx = 600
    ny = 300
    ns = 10
    img = ["P3\n", str(nx)+" "+str(ny), "\n255\n"]

    world = random_scene()

    lookfrom = vec3(13.0, 2.0, 3.0)
    lookat = vec3(0.0, 0.0, 0.0)
    dist_to_focus = 10.0
    aperture = 0.1

    cam = camera(20, nx, ny, lookfrom, lookat, vec3(0,1,0), aperture, dist_to_focus)

    for j in range(ny-1, -1, -1):
        for i in range(nx):
            col = vec3(0.0, 0.0, 0.0)
            for s in range(ns):
                u = (i + random.random())/nx
                v = (j + random.random())/ny

                r = cam.get_ray(u, v)
                col += color(r, world, 0)
            
            ic = vec3(math.sqrt(col.x/ns), math.sqrt(col.y/ns), math.sqrt(col.z/ns)) * 255.99
            pc = " ".join(map(lambda x: str(int(x)), ic.make_list())) + "\n"
            img.append(pc)

    f = open("raytracing.ppm", 'w')
    f.writelines(img)
    f.close()
    
raytracing()