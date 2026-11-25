"""Draw an RNN cell unrolled across four time steps."""
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(9,3)); ax.axis('off')
for i in range(4):
 x=i*2.1; ax.text(x,0,'xₜ',ha='center',bbox=dict(boxstyle='round',fc='lightblue')); ax.text(x,1,'hₜ',ha='center',bbox=dict(boxstyle='round',fc='wheat')); ax.text(x,2,'yₜ',ha='center',bbox=dict(boxstyle='round',fc='lightgreen'))
 ax.annotate('',(x,.75),(x,.2),arrowprops=dict(arrowstyle='->')); ax.annotate('',(x,1.75),(x,1.25),arrowprops=dict(arrowstyle='->'))
 if i<3: ax.annotate('',(x+1.55,1),(x+.4,1),arrowprops=dict(arrowstyle='->'))
ax.set_xlim(-.7,7.8); ax.set_ylim(-.5,2.5); ax.set_title('RNN unrolled over time (shared weights)'); plt.tight_layout(); plt.show()
