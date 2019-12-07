## using graphviz 
#!apt-get -qq install -y graphviz && pip install -q pydot
import pydot
### 아래 있는것은 필수는 아닌데, 가끔 에러가 생길 때가 있어서, 그냥 같이 해줌. 
#!apt-get install graphviz libgraphviz-dev pkg-config
#!pip install pygraphviz
import pygraphviz as pgv
from IPython.display import Image
from random import randint

G = pgv.AGraph()
centernode = 'chogook'
nodelist=['chomin','president moon','Trump','controversy','law','professor','politics']
G.add_nodes_from(nodelist)
G.add_node(centernode) # adds center node

# Setting node attributes that are common for all nodes 
G.node_attr['style']='filled'
G.node_attr['color']='#FFFFFF'
G.node_attr['fontcolor']= 'red'
G.edge_attr['style'] = 'invis'

# similarity dictionary
sim_dict = {'chomin':9,'president moon':5,'Trump':2,'controversy':8,'law':1,'professor':10,'politics':7}

# Creating and setting node attributes that vary for each node (using a for loop)
center = G.get_node(centernode)
center.attr['style']='filled'
center.attr['shape']='circle'
center.attr['color']='lightgray'
center.attr['fontsize']='50'
center.attr['fontcolor']='black'

for i in nodelist:
 G.add_edge(centernode,i)
 n = G.get_node(i)
 n.attr['fontcolor']="#%2x0000"%(sim_dict[i]) # random?
 n.attr['height']="%s"%(sim_dict[i]*0.3)
 n.attr['fontsize']="%s"%(sim_dict[i]*8 + 30) # 관련성에 비례하게
 n.attr['fontcolor']="#%2x"%(randint(0,16777215+1)) # random?


G.draw('viz.png',prog="circo") # This creates a .png file in the local directory. Displayed below.

Image('viz.png', width=700) # The Graph visualization we created above.
