# author: zhaofeng-shu33
# file: load 3d mesh files(ply) and save them to pictures from different angles
import os
from math import cos, sin, pi
import open3d
ANGLE_SPLIT = 10
def custom_draw_geometry(pcd, person_id, pose_id):
    # The following code achieves the same effect as:
    # draw_geometries([pcd])
    vis = open3d.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json("renderoption.json")    
    for i in range(ANGLE_SPLIT):
        angle = 2*pi/ANGLE_SPLIT
        trans = [[cos(angle), 0, -sin(angle),  0.0],
                 [0,          1, 0,            0.0],
                 [sin(angle), 0, cos(angle),   0.0],
                 [0,          0, 0,            1.0]]    
        pcd.transform(trans)                    
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()                        
        angle_id = i
        output_fileName = "%d_%d_%d.jpg" % (person_id, pose_id, angle_id)        
        vis.capture_screen_image(os.path.join('output', output_fileName))
    vis.destroy_window()
    
if __name__ == '__main__':
    for person_id in range(10):
        for pose_id in range(10):
            scan_id = person_id * 10 + pose_id            
            fileName = os.path.join('scans', 'tr_scan_%03d.ply' % scan_id)
            mesh = open3d.read_triangle_mesh(fileName)
            custom_draw_geometry(mesh, person_id, pose_id)