from manim import *

# For Windows users, make sure you have installed the latest version of ffmpeg and added it to your PATH.
# Also, make sure that Latex is installed and added to your PATH.

# To run this script, use the command:
# manim -pql matrix_vectors.py TermDocumentVectors2D
# manim -pql matrix_vectors.py TermDocumentVectors3D

class TermDocumentVectors3D(ThreeDScene):
    def construct(self):
        # Axes for 3D term space
        axes = ThreeDAxes(
            x_range=[0, 4],
            y_range=[0, 4],
            z_range=[0, 5],
            x_length=5, y_length=5, z_length=5
        )

        labels = axes.get_axis_labels(x_label="cat", y_label="dog", z_label="mouse")
        # self.set_camera_orientation(phi=70 * DEGREES, theta=45 * DEGREES)

        self.play(Create(axes), Write(labels))

        # Document vectors from term-document matrix
        doc1 = [1, 0, 4]
        doc2 = [2, 3, 0]

        v1 = Vector(axes.c2p(*doc1), color=BLUE)
        v2 = Vector(axes.c2p(*doc2), color=GREEN)

        l1 = MathTex("D_1", color=BLUE).next_to(v1.get_end(), UP)
        l2 = MathTex("D_2", color=GREEN).next_to(v2.get_end(), RIGHT)

        self.play(GrowArrow(v1), Write(l1))
        self.play(GrowArrow(v2), Write(l2))

        self.wait(2)

class TermTermVectors2D(Scene):
    def construct(self):
        # 2D term-term similarity matrix: each row as a vector
        axes = Axes(
            x_range=[0, 1.2],
            y_range=[0, 0.6],
            x_length=6,
            y_length=3,
            axis_config={"include_tip": True}
        )
        labels = axes.get_axis_labels(x_label="dog", y_label="mouse")

        self.play(Create(axes), Write(labels))

        # Term vectors: cat, dog, mouse (projected on 2D)
        cat_vec = [0.2, 0.4]
        dog_vec = [1.0, 0.3]
        mouse_vec = [0.3, 1.0]

        v_cat = Vector(axes.c2p(*cat_vec), color=RED)
        v_dog = Vector(axes.c2p(*dog_vec), color=ORANGE)
        v_mouse = Vector(axes.c2p(*mouse_vec), color=PURPLE)

        l_cat = MathTex("cat", color=RED).next_to(v_cat.get_end(), LEFT)
        l_dog = MathTex("dog", color=ORANGE).next_to(v_dog.get_end(), RIGHT)
        l_mouse = MathTex("mouse", color=PURPLE).next_to(v_mouse.get_end(), UP)

        self.play(GrowArrow(v_cat), Write(l_cat))
        self.play(GrowArrow(v_dog), Write(l_dog))
        self.play(GrowArrow(v_mouse), Write(l_mouse))

        self.wait(2)