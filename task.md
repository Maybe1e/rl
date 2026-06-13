Project Brief: Applied Machine Learning 25/26
Reinforcement and Imitation Learning for 
Robotic Manipulation in Creative 
Expression
Imagine you have a robotic mechanism you can teach how to move. The robot 
can be of any shape. Your task is to use reinforcement or imitation learning 
to teach it how to perform a task from the domain of creativity and arts.
1. Project Description
In this project you will be working with machine learning applied to the robotic manipulation 
task that achieves a creative outcome. There are two options, A and B, for completing this 
project. For either option you can develop a simulation in Unity or Unreal Engine. 
Reinforcement Learning for Expressive Robot Movement
Option A: Reinforcement Learning (RL)
In the option A you are tasked with the interpretation of the manipulation task as an expressive 
behaviour. Your aim is to train, test and demo a reinforcement learning agent that learns an 
expressive movement pattern based on the rewards and completitions* you design and 
implement as you RL policy in either Unreal Engine or Unity.
The robot should learn to generate a movement pattern that satisfies a creative objective. 
Examples are a) drawing a visual pattern with robotic arm, b) producing a dance-like motion 
sequence with a humanoid, c) moving through a space while maintaining fluidity and 
gracefulness with the quadruped.
*In reinforcement learning, training is a sequence of observations, actions, and rewards. In 
addition to normal simulation ending conditions (e.g. completing the task), additional logic can be 
added to end training early.
Option B: Imitation Learning (IL)
Option B of the project intends to accomplish the same objective as option A with the difference 
being that you will use imitation learning* instead of reward engineering. Your aim is to train, 
test and demo an agent that learns an expressive movement pattern by observing and 
reproducing demonstrations you provide.
You will collect demonstration data by manually controlling the robot (via teleoperation, 
kinesthetic teaching, or scripted reference trajectories) performing the desired creative 
behaviour or by simulation such data with code. The agent then learns to replicate and 
generalise from these demonstrations using Behavioural Cloning or DAgger, implemented in 
either Unreal Engine or Unity.
The robot should learn to generate a movement pattern that satisfies a creative objective. 
Examples are a) drawing a visual pattern with a robotic arm by tracing demonstrations, b) 
producing a dance-like motion sequence with a humanoid from recorded movement, c) moving 
through a space while maintaining fluidity and gracefulness with a quadruped guided by 
reference trajectories. 
*In imitation learning, training is a sequence of state-action pairs extracted from expert 
demonstrations. Rather than designing a reward function, you design the demonstrations deciding 
what constitutes the target expressive behaviour by showing it.
1.1 Objectives
Communication: Your project should feature the visual documentation of the system 
developed. This should be done within the report by the inclusion of the infographics. 
Please refer to the premier ML publications in the field for the types of the graphs used 
e.g. CVF Computer Vision and Pattern Recognition Conference (CVPR)
Assessment criteria: Your porject should provide the evidence for you meeting the 
following learning objectives (LO)
LO1 Demonstrate comprehensive understanding of fundamental machine learning 
concepts and algorithms (Communication, Knowledge), 
LO2 Critically assess the selection of ML approaches for specific robotic applications, 
considering computational and hardware constraints. (Enquiry),
LO3 Work with complex adaptive robots in simulated environments (Enquiry) 
Implement and analyse robotic solutions, demonstrating the ability to integrate ML 
algorithms with hardware considerations. (Process, Realisation).
2. Submissions
2.1 Final Report
A report showing the process of the project development and your independent inquiry is as 
important as the code files. It should clearly document your work.
The report should include:
Abstract: Summarise the key elements of your project, including its purpose, key findings, and 
contributions. This section should provide a concise overview that highlights the unique aspects 
of your work.
CCS Concepts: Provide the ACM Computing Classification System (CCS) concepts that are 
relevant to your project. Use appropriate keywords and categories that reflect your project's 
focus on topics.
Keywords: Include a list of keywords that represent the core topics of your project.
Report Sections:
1. Introduction: Provide an overview of the concept and objectives of your project. 
Explain what motivated the idea and how it fits within the broader context of the 
applications of machine learning (ML) in robotics and ML simulation.
2. Background: Describe the research that informs your project. This could include a 
review of related works e.g. relevant projects, technologies, artists, or research papers 
that influenced your approach. Establish how your project builds upon or diverges from 
these inspirations.
3. Creative Brief: This should include problem description and your creative process.
a. Concept Description: Describe what pattern(s)/symbols your CNN will 
recognise and how this would inform the user's ability to control and 
communicate with the robot. Please include the dataset description and sample 
images used for training/testing.
b. Contextual Research: Situate your project within the broader field of computer 
vision. Mention relevant research papers, existing applications, or technologies 
that have influenced your approach and how your work builds on these ideas.
4. Technical Implementation: Detail the technical aspects of your project.
a. RL Policy: Explain how you have chosen the observations, actions and rewards 
for you RL training. 
b. RL/IL Algorithm: Describe what algorithm you have chosen for your learning 
process e.g. PPO, actor critic. 
c. GitHub: Please save your training environment as a Docker image and submit 
this to a GitHub repository. Include a clickable link to the repository in your 
report. 
5. Results: Present your process of testing and demo of the system in a 1 minute video.
Highlight what worked well, what challenges you encountered, and how you iterated 
upon technical issues. You don’t need to include the voiceover in the video as long as the 
results are sufficiently described in the report. 
6. Reflection: Critically evaluate the outcomes of your project. Reflect on how effectively 
the project meets your original objectives. This section should also mention any 
limitations of your project and suggest areas for future development.
7. References: List all resources cited in your report, including articles, books, 
documentation, and other projects that informed your work. Use the ACM citation 
format.
Report should be delivered in the format following ACM Conference Proceedings Primary 
Article Template
• Overleaf: https://www.overleaf.com/latex/templates/acm-conference￾proceedings-primary-article-template/wbvnghjbzwpc
2.2 ML Training Files
You should submit only the final training outcomes (saved as ONNX ) of your simulation 
without other project files. Please include the link to a complete GitHub repository that will 
contain a Docker image of you training environment in the report under the technical 
implementation section. 
2.3 Demo
Please submit a 1 minute video showing your testing process and the demo of 
your final simulation running.
2.3 Reference AI in your work
Please ensure to clearly state and describe how you have used the AI tools for your project. Like 
any other sources you may consider using in assessments such as a journal, book or database, 
you must properly reference (cite) AI in your work. The Cite Them Right website provides 
guidance on how to reference generative AI in work. You can also refer to UAL's Student 
Guide to Generative AI.


翻译一下中文