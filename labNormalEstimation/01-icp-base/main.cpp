#include <iostream>
#include "GL/gl3w.h"
#include <GLFW/glfw3.h>
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include "imgui/imgui_internal.h"
#include "Application.h"


using namespace ImGui;


#define VIEWPORT_INIT_WIDTH 640
#define VIEWPORT_INIT_HEIGHT 480

#define TARGET_FRAMERATE 60.0f


void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if(!ImGui::GetIO().WantCaptureKeyboard)
	{
		ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
		if (action == GLFW_PRESS)
			Application::instance().keyPressed(key);
		else if (action == GLFW_RELEASE)
			Application::instance().keyReleased(key);
	}
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
	ImGui_ImplGlfw_CursorPosCallback(window, xpos, ypos);
	if(!ImGui::GetIO().WantCaptureMouse)
		Application::instance().mouseMove(int(xpos), int(ypos));
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	int buttonId;
	
	ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
	switch(button)
	{
	case GLFW_MOUSE_BUTTON_LEFT:
		buttonId = 0;
		break;
	case GLFW_MOUSE_BUTTON_RIGHT:
		buttonId = 1;
		break;
	case GLFW_MOUSE_BUTTON_MIDDLE:
		buttonId = 2;
		break;
	}

	if(action == GLFW_PRESS)
		Application::instance().mousePress(buttonId);
	else if(action == GLFW_RELEASE)
		Application::instance().mouseRelease(buttonId);

}

void window_size_callback(GLFWwindow* window, int width, int height)
{
	SetNextWindowSize(ImVec2(width, height));
	Application::instance().resize(width, height);
}

void gui_display()
{
	Application::instance().render_gui();
}

void render()
{
	Application::instance().render();

	// Start the Dear ImGui frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	
	gui_display();

	// Rendering
	ImGui::EndFrame();
	ImGui::Render();
	ImGuiIO& io = ImGui::GetIO();
	glViewport(0, 0, (GLsizei)io.DisplaySize.x, (GLsizei)io.DisplaySize.y);
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}


int main(int argc, char *argv[])
{
	GLFWwindow* window;
	double timePerFrame = 1.f / TARGET_FRAMERATE, timePreviousFrame, currentTime;

	/* Initialize the library */
	if (!glfwInit())
		return -1;

	/* Create a windowed mode window and its OpenGL context */
	window = glfwCreateWindow(VIEWPORT_INIT_WIDTH, VIEWPORT_INIT_HEIGHT, "01 - Normals & ICP", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	/* Set window initial position */
	glfwSetWindowPos(window, 100, 100);
	/* Make the window's context current */
	glfwMakeContextCurrent(window);

	/* Set callbacks */
	glfwSetKeyCallback(window, key_callback);
	glfwSetCursorPosCallback(window, cursor_position_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetWindowSizeCallback(window, window_size_callback);

	// GL3W will take care of OpenGL extension functions
	gl3wInit();
	
	/* Init step of the game loop */
	Application::instance().init();
	Application::instance().resize(VIEWPORT_INIT_WIDTH, VIEWPORT_INIT_HEIGHT);
	if(argc == 3)
		Application::instance().loadScans(argv[1], argv[2]);
	else
	{
		cout << "Wrong parameters!" << endl << endl;
		cout << "Usage:" << endl << endl;
		cout << argv[0] << " <Point Cloud 1> <Point Cloud 2>" << endl;
		
		return -1;
	}
	timePreviousFrame = glfwGetTime();
	
	// Setup Dear ImGui context
	ImGui::CreateContext();
	//ImGuiIO& io = ImGui::GetIO(); (void)io;
	
	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsClassic();

	// Setup Platform/Renderer bindings
	if (!ImGui_ImplGlfw_InitForOpenGL(window, false))
		return -1;
	if (!ImGui_ImplOpenGL3_Init())
		return -1;
	
	/* Loop until the user closes the window */
	while (!glfwWindowShouldClose(window))
	{
		currentTime = glfwGetTime();
		if (currentTime - timePreviousFrame >= timePerFrame)
		{
			/* Update & render steps of the game loop */
			if(!Application::instance().update(int(1000.0f * (currentTime - timePreviousFrame))))
				glfwSetWindowShouldClose(window, GLFW_TRUE);
			render();
			timePreviousFrame = currentTime;

			/* Swap front and back buffers */
			glfwSwapBuffers(window);
		}

		/* Poll for and process events */
		glfwPollEvents();
	}

	// Shutdown GLFW
	glfwTerminate();

	// Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	
	return 0;
}




