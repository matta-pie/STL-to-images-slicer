# GLOBAL VARIABLES
# tuning general parameters
scale = 1000
image_padding = 100
y_slice_start = 0
y_slice_step = 0.01
y_slice_num = round(1/y_slice_step)
line_width = 5
salto = 3
model = pywavefront.Wavefront('bladdermodel_cm4.obj', collect_faces=True) #select which model you want to perform the strategy with

border = []
border2 = []
path = []
angle_max = 120 #curvature radius max angle 
val = 1

# where is the face w.r.t. to y value
def facePosition(face, y):
    # I save the vertexes of a face 
    a = model.vertices[face[0]]
    b = model.vertices[face[1]]
    c = model.vertices[face[2]]

    if(a[1] >= y and b[1] >= y and c[1] >= y):
        return "above"
    elif(a[1] <= y and b[1] <= y and c[1] <= y):
        return "below"
    elif(a[1] == y and b[1] == y and c[1] == y):
        return "planar"
    else:
        return "intersect"

# from RosettaCode
def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-12):
    norm = np.linalg.norm(rayDirection)
    rayDirection = rayDirection/norm

    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        return np.array([])
    else:
        w = rayPoint - planePoint
        si = -planeNormal.dot(w) / ndotu
        Psi = w + si * rayDirection + planePoint
        return Psi

def drawIntersect(face, y, draw):
    planeNormal = np.array([0,1,0])
    planePoint = np.array([0,y,0])
    a = np.array(model.vertices[face[0]]) #vertex 1 
    b = np.array(model.vertices[face[1]]) #vertex 2
    c = np.array(model.vertices[face[2]]) #vertex 3

    # Find points of intersection between line and plane. 
    # p1 and p2 are normalized [0,1]
  
    if ((a[1] > y and b[1] > y and c[1] < y) or (a[1] < y and b[1] < y and c[1] > y)):
        p1 = LinePlaneCollision(planeNormal, planePoint, np.subtract(a, c), a)
        p2 = LinePlaneCollision(planeNormal, planePoint, np.subtract(b, c), b)
    elif ((a[1] > y and c[1] > y and b[1] < y) or (a[1] < y and c[1] < y and b[1] > y)):
        p1 = LinePlaneCollision(planeNormal, planePoint, np.subtract(a, b), a)
        p2 = LinePlaneCollision(planeNormal, planePoint, np.subtract(c, b), c)
    elif ((b[1] > y and c[1] > y and a[1] < y) or (b[1] < y and c[1] < y and a[1] > y)):
        p1 = LinePlaneCollision(planeNormal, planePoint, np.subtract(b, a), b)
        p2 = LinePlaneCollision(planeNormal, planePoint, np.subtract(c, a), c)

    # I take the 3D intersection point and scale to fit the image
    x1 = (p1[0] * scale) + (image_padding/2) 
    z1 = (p1[2] * scale) + (image_padding/2)
    x2 = (p2[0] * scale) + (image_padding/2)
    z2 = (p2[2] * scale) + (image_padding/2)
    

    # draw the line between the two points found, to have a complete draw
    draw.line((x1, z1, x2, z2), fill=128, width=line_width)

    # Alternatively, I can draw only the points 
    #draw.point([x1, z1, x2, z2], fill = 128)

    return (p1, p2)


def Borders(p1): #saving the points defining the borders in an array 
    border =[]

    #x_1 = p1[0]*x_max-x_offset  #offset is negative
    #è come scrivere x1 = p1[0]*(model_box[1][0]-model_box[0][0]) + model_box[0][0] 

    ##  To have them in the model
    x1 = p1[0]*(model_box[1][0]-model_box[0][0]) + model_box[0][0]
    y1 = p1[1]*(model_box[1][1]-model_box[0][1]) + model_box[0][1]
    z1 = p1[2]*(model_box[1][2]-model_box[0][2]) + model_box[0][2]

    ## Scaling them for the image drawing 
    x_1 = (p1[0] * scale) + (image_padding/2)
    y_1 = (p1[1] * scale) + (image_padding/2)
    z_1 = (p1[2] * scale) + (image_padding/2)
    
    border_1 = x1, y1, z1 # model coordinate 
    border_2 = x_1, y_1, z_1 # image coordinate 
    border_3 = p1[0], p1[1], p1[2] # normalized 

    return(border_2)  #select which array you want 
    
    
# check all vertexes to define the size of the model 
model_box = (model.vertices[0], model.vertices[0]) # first couple of vertexes 
for vertex in model.vertices:
    min_v = [min(model_box[0][i], vertex[i]) for i in range(3)] # check everything and store the triplet corresponding to min x, min y, min z
    max_v = [max(model_box[1][i], vertex[i]) for i in range(3)] # check and store in the second position the triplet max x, max y, max z
    model_box = (min_v, max_v) # firs three : min x, min y, min z
                               # second three: max x, max y, max z
                               
# find the offset of the model
# I use a negative offset so that if the offset is negative it sums up,
# if it's a positive value I'll have a subtracion

x_offset = -model_box[0][0]
y_offset = -model_box[0][1]
z_offset = -model_box[0][2]

x_max = model_box[1][0] + x_offset ## it'sthe difference between the max value of the model and the offset (= size of of the model)
y_max = model_box[1][1] + y_offset 
z_max = model_box[1][2] + z_offset

print("offset x: " + str(x_offset))
print("offset y: " + str(y_offset))
print("offset z: " + str(y_offset))

print("max x: " + str(x_max))
print("max y: " + str(y_max))
print("max z: " + str(y_max))

y_slice_index = y_slice_start

for idx, vertex in enumerate(model.vertices):

    # I convert the touple in list as touples are not changable in python
    # Translating every vertex of a value equal to the offset and then
    # making a linear interopoalation to have a normalized model ->[0,1]

    model.vertices[idx] = list(vertex)
    model.vertices[idx][0] = np.interp(vertex[0] + x_offset,[0,x_max],[0,1])
    model.vertices[idx][1] = np.interp(vertex[1] + y_offset,[0,y_max],[0,1])
    model.vertices[idx][2] = np.interp(vertex[2] + z_offset,[0,z_max],[0,1])

# the obj file could have more than one model, loop each of them 
counter = 0
while y_slice_index < 1:
    img = Image.new('RGB', (scale + image_padding, scale + image_padding), color = 'white')
    draw = ImageDraw.Draw(img)
    for mesh in model.mesh_list:
        #for each face of each model 
        for face in mesh.faces:
            result = facePosition(face, y_slice_index)
            if(result == "intersect"):
               p1, p2 = drawIntersect(face, y_slice_index, draw)
               bord = Borders(p1)
               border.append(bord)
               bord = Borders(p2)
               border.append(bord) #saving al the borders coords
                #it's a list
    img.save('output/' + str(counter) + '.png')
    counter = counter + 1
    y_slice_index = y_slice_index + y_slice_step
    y_slice_index = round(y_slice_index, 5)

