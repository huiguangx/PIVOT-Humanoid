#!/usr/bin/env python3
import argparse

import numpy as np
import onnxruntime as ort
import torch


class StatelessPolicy(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.memory = torch.nn.LSTM(41, 64, 1)
        self.memory.load_state_dict(source.memory.state_dict())
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ELU(),
            torch.nn.Linear(32, 10),
        )
        self.actor.load_state_dict(source.actor.state_dict())

    def forward(self, policy, hidden_state, cell_state):
        output, (next_hidden, next_cell) = self.memory(
            policy.unsqueeze(0), (hidden_state, cell_state)
        )
        action = self.actor(output.squeeze(0))
        return action, next_hidden, next_cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()

    torch.manual_seed(7)
    source = torch.jit.load(args.source, map_location="cpu").eval()
    policy = StatelessPolicy(source).eval()
    observation = torch.zeros(1, 41)
    hidden = torch.zeros(1, 1, 64)
    cell = torch.zeros(1, 1, 64)

    torch.onnx.export(
        policy,
        (observation, hidden, cell),
        args.destination,
        input_names=["policy", "hidden_state", "cell_state"],
        output_names=["action", "next_hidden_state", "next_cell_state"],
        opset_version=17,
        dynamo=False,
    )

    session = ort.InferenceSession(args.destination, providers=["CPUExecutionProvider"])
    max_error = 0.0
    for _ in range(8):
        observation = torch.randn(1, 41)
        with torch.no_grad():
            expected = policy(observation, hidden, cell)
        actual = session.run(None, {
            "policy": observation.numpy(),
            "hidden_state": hidden.numpy(),
            "cell_state": cell.numpy(),
        })
        max_error = max(max_error, *(np.max(np.abs(a - b.numpy())) for a, b in zip(actual, expected)))
        hidden, cell = expected[1], expected[2]

    assert max_error < 1e-4, f"max_abs_error={max_error}"
    print(f"inputs: policy[1,41], hidden_state[1,1,64], cell_state[1,1,64]")
    print(f"outputs: action[1,10], next_hidden_state[1,1,64], next_cell_state[1,1,64]")
    print(f"max_abs_error={max_error:.8g}")


if __name__ == "__main__":
    main()
