from roboticstoolbox.models import Mico
import numpy as np
from rtb_prompt import SYSTEM_PROMPT
from ipython_secrets import get_secret
from openai import OpenAI

KEY = get_secret("OPENAI_API_KEY")
openai = OpenAI(api_key=KEY)

chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

class MaxBot(Mico):
    def __init__(self):
        # Initialize the robot model
        # Call the constructor of the parent class (Mico)
        super().__init__()

        # Start in vertical ‘READY’ configuration
        self.q = self.qr

    def set_joint_angles(self, waist, shoulder, elbow):
        # Angles should be specified in degrees
        self.q[0] = np.deg2rad(waist)
        self.q[1] = np.deg2rad(shoulder)
        self.q[2] = np.deg2rad(elbow)

    def get_hand_orientation(self):
        # Get the current orientation of the robot's end-effector
        return np.rad2deg(self.q[3])

def copilot(prompt):
    chat_history.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo", messages=chat_history, temperature=0
    )

    chat_history.append(
        {
            "role": "assistant",
            "content": completion.choices[0].message.content,
        }
    )

    return chat_history[-1]["content"]