#ifndef GRAPHICSMODE_H
#define GRAPHICSMODE_H
#define GL_H
#define GL_GLEXT_PROTOTYPES
#define GRAPHICS_H
#endif

#define cimg_display 0

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <GL/gl.h>
#include <GL/freeglut.h>
#include <cuda.h>
#include <cudaGL.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <CImg.h>




using namespace std;
using namespace cimg_library;

#define REFRESH_DELAY 10 
#define DELTA 10
#define W 512
#define H 512
#define TX 32
#define TY 32
#define RAD 1

static const char *windname = "Image Sharpening";

//Graphics Resource objects
GLuint pbo = 0;
GLuint tex = 0;
struct cudaGraphicsResource *cuda_pbo_resource;
struct uchar4;
struct float4;
struct int3 loc = {W/2, H/2, 1};

// Parameters initialization
int sys = 2;
float param = 0.25f;
GLfloat angle1 = 0.0f;
GLfloat angle2 = 0.0f;
float g_fAnim = 0.0;
int mouse_old_x; 
int mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0;
float rotate_y = 0.0;
float translate_z = -3.0;
float scale = 500;








int divUp(int a, int b) { return (a + b - 1) / b; }

__device__
unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

__device__
int idxClip(int idx, int idxMax) {
    return idx > (idxMax-1) ? (idxMax-1) : (idx < 0 ? 0 : idx);
}

__device__
int flatten(int col, int row, int width, int height) {
    return idxClip(col, width) + idxClip(row, height)*width;
}

__global__
void sharpenKernel(uchar4 *d_out, const uchar4 *d_in, const float *d_filter, int w, int h) {
    const int c = threadIdx.x + blockDim.x * blockIdx.x;
    const int r = threadIdx.y + blockDim.y * blockIdx.y;
    
    if ((c >= w) || (r >= h)) return;
    
    
    const int i = flatten(c, r, w, h);
    const int fltSz = 2*RAD + 1;
    float rgb[3] = {0.f, 0.f, 0.f};
    
    for (int rd = -RAD; rd <= RAD; ++rd) {
        for (int cd = -RAD; cd <= RAD; ++cd) {
            int imgIdx = flatten(c + cd, r + rd, w, h);
            int fltIdx = flatten(RAD + cd, RAD + rd, fltSz, fltSz);
            uchar4 color = d_in[imgIdx];
            float weight = d_filter[fltIdx];
            rgb[0] += weight*color.x;
            rgb[1] += weight*color.y;
            rgb[2] += weight*color.z;
        }
    }
    
    d_out[i].x = clip(rgb[0]);
    d_out[i].y = clip(rgb[1]);
    d_out[i].z = clip(rgb[2]);
}


// Cuda run model
 void cudarun() {
    //Image
    CImg<unsigned char> img ("src2.bmp");
    
    //input data
    uchar4 *arr=(uchar4*)malloc(W*H*sizeof(uchar4));
    
    // Copy data to array
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c){
            arr[r*W + c].x = img(c,r, 0);
            arr[r*W + c].y = img(c,r, 0);
            arr[r*W + c].z = img(c,r, 0);
            arr[r*W + c].w = 0;
        }
    }
    
    //device storage 
    const int fltSz = 2 * RAD + 1;
    const float filter[9] = {1.0/16, 2.0/16, 1.0/16,
			2.0/16, 4.0/16, 2.0/16,
			1.0/16, 2.0/16, 1.0/16};
    
    uchar4 *d_in = 0, *d_out = 0;
    float *d_filter = 0;
    
    cudaMalloc(&d_in, W*H*sizeof(uchar4));
    cudaMemcpy(d_in, arr, W*H*sizeof(uchar4), cudaMemcpyHostToDevice);
    cudaMalloc(&d_out, W*H*sizeof(uchar4));
    cudaMalloc(&d_filter, fltSz*fltSz*sizeof(float));
    cudaMemcpy(d_filter, filter, fltSz*fltSz*sizeof(float),cudaMemcpyHostToDevice);
    
       
    //Graphics resources map
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL,cuda_pbo_resource);    
    
    //kernelLauncher   
    const dim3 blockSize(TX, TY);
    const dim3 gridSize(divUp(W, blockSize.x), divUp(H, blockSize.y)); 
    
    //const dim3 blockSize(TX, TY);
    //const dim3 gridSize = dim3((W + TX - 1)/TX, (H + TY - 1)/TY);
    
    sharpenKernel<<<gridSize, blockSize>>>(d_out, d_in, d_filter, W, H);  
    cudaDeviceSynchronize();
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
    
    //SAVE IMAGE: Copy from array to CImg data
    cudaMemcpy(arr, d_out, W*H*sizeof(uchar4), cudaMemcpyDeviceToHost);
    for (int r =0; r < H; ++r) {
        for (int c= 0; c < W; ++c) {
            img(c,r, 0) = arr[r*W + c].x;
            img(c,r, 1) = arr[r*W + c].y;
            img(c,r, 2) = arr[r*W + c].z;
        }
    }
    img.save_bmp("out.bmp");

 }

 
 
 
// Display model
static void display(){
    
    cudarun();
    glClearColor(0.0,0.34,0.46,1.0);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHT1);
    glShadeModel(GL_SMOOTH);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_NORMALIZE);
    
    
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA,GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);    
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    
    //Textue map
    glEnable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE_2D);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glBegin(GL_QUADS);
    	glTexCoord2f(0.0f, 0.0f); glVertex2f(0,0);
    	glTexCoord2f(0.0f, 1.0f); glVertex2f(0,H);
    	glTexCoord2f(1.0f, 1.0f); glVertex2f(W,H);
    	glTexCoord2f(1.0f, 0.0f); glVertex2f(W,0);
    glEnd();
    glPopMatrix();
    

    glFlush();
    glDepthFunc(GL_LEQUAL); 
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glutSwapBuffers();  
    g_fAnim += 0.05f;
    glDisable(GL_TEXTURE_2D);
    
}


// Reshape window
static void reshape(int w, int h){
    glViewport(0, 0, (GLsizei) w, (GLsizei) h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, W, H, 0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

// Pixel Buffer generate
static void initPixelBuffer() {
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 4*W*H*sizeof(GLubyte), 0, GL_STREAM_DRAW);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,cudaGraphicsMapFlagsWriteDiscard);
}


// Handler for animation
static void animate(void){
     angle1  = 0.01f*glutGet(GLUT_ELAPSED_TIME);
     angle2 =  0.01f*glutGet(GLUT_ELAPSED_TIME);
     glutPostRedisplay();

}

// Handler for timer event
static void timerEvent(int value){
    if (glutGetWindow()) {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    }
}

// Free Buffer and Texture
static void exitfunc() {
    if (pbo) {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }
}

// Handler for mous event
static void mouskey(int button,int state, int x, int y){
    if (state == GLUT_DOWN)    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)    {
        mouse_buttons = 0;
    }
    mouse_old_x = x;
    mouse_old_y = y;
        
 }


// Handler for key event
static void keyboard(unsigned char key, int x, int y) {
  if(x==0||y==0) return;
    switch (key){
        case (27) :
            if (key==27||key=='q'||key=='Q')
                exit(EXIT_SUCCESS);
            else
                glutDestroyWindow(glutGetWindow());
                return;        
    }
}


static void motion(int x, int y){
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);
    if (mouse_buttons & 1){
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4){
        translate_z += dy * 0.01f;
    }
    mouse_old_x = x;
    mouse_old_y = y;
}


int main(int argc, char** argv) {
    cudaDeviceProp  prop;
    cudaGetDeviceProperties(&prop, 0);
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE|GLUT_DEPTH);
    glutInitWindowSize(W, H);
    glutInitWindowPosition(200, 200);
    glutCreateWindow(windname);     
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    initPixelBuffer();
    glutIdleFunc(animate);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    glutMouseFunc(mouskey);
    glutMotionFunc(motion);
    glutKeyboardFunc(keyboard);
    glutMainLoop();
    atexit(exitfunc);
    return 0;
}
