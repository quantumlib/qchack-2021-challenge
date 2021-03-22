# Copyright 2021 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import traceback

import cirq
from attr import dataclass
from typing import Callable, List, Optional
import numpy as np
from cirq.testing import assert_allclose_up_to_global_phase


@dataclass
class Subtask:
    subtask_name: str
    max_score: int
    evaluate: Callable


@dataclass
class JudgeLogEntry:
    task: str
    actual_score: int = 0
    max_score: int = 0
    msgs: str = ""

    def title(self):
        pad = "/\\" * int((50 - len(self.task)) / 2)
        return f"{pad} [ {self.task} ] {pad}"

    def __str__(self):
        return f"""{self.msgs}
Result: {self.actual_score:.2f} / {self.max_score}"""


@dataclass
class JudgeLog:
    entries: List[JudgeLogEntry] = []

    def results(self):
        total = sum(e.actual_score for e in self.entries)
        total_max = sum(e.max_score for e in self.entries)
        # lines = "\n".join(str(e) for e in self.entries)
        return f"""
{"=" * 100}
Total score: {total:.2f} / {total_max} points!
"""


def score_input(
    matrix_to_sycamore_operations: Callable,
    input: np.ndarray,
    result: JudgeLogEntry,
    multiplier: int,
    n_qubits: int,
    min_two_qubit,
):
    # see Shende et al.
    theoretical_lower_bound = int(1 / 4 * (4 ** n_qubits - 3 * n_qubits - 1))
    generic_lower_bound = min(theoretical_lower_bound, min_two_qubit)

    # max score settings
    max_trace_distance_score = multiplier
    result.max_score += max_trace_distance_score

    max_two_qubit_gate_count_score = multiplier * 2
    result.max_score += max_two_qubit_gate_count_score

    max_sycamore_score = multiplier
    result.max_score += max_sycamore_score

    # prepare qubits
    if n_qubits < 4:
        qs = cirq.GridQubit.rect(1, n_qubits, 3, 3)
    elif int(np.sqrt(n_qubits)) ** 2 == n_qubits:
        qs = cirq.GridQubit.square(int(np.sqrt(n_qubits)), 3, 3)
    elif n_qubits % 2 == 0:
        qs = cirq.GridQubit.rect(2, int(n_qubits / 2), 3, 3)
    else:
        qs = cirq.GridQubit.rect(2, int((n_qubits + 1) / 2), 3, 3)[:-1]

    # an executing method is mandatory
    skipped = False
    result.msgs += f"\nexecuting method (0 pts): "
    try:
        response, ancillae = matrix_to_sycamore_operations(qs, input)
        if response == NotImplemented:
            skipped = True
            result.msgs += f"✔ [0 pts]"
        elif response != NotImplemented:
            response_circuit = cirq.Circuit(response)

            total_qubit_count = len(qs) + len(ancillae)
            assert (
                total_qubit_count <= 10
            ), f"Number of total qubits (target + ancilla) can't be larger than 10! Response has {total_qubit_count}."

            response_unitary = response_circuit.unitary(
                qubit_order=qs + ancillae, qubits_that_should_be_present=qs + ancillae
            )
            expected_unitary = cirq.kron(input, np.eye(2 ** len(ancillae)))
            result.msgs += f"✔ [0 pts]"
        failed = False
    except BaseException as ex:
        response = ex
        response_tb = traceback.format_exc()
        failed = True
        result.msgs += (
            f"✘\n {type(response)}: {str(response)[:50]}" f"\n {response_tb[:500]}"
        )

    #####
    # scoring functions for each feature
    #####

    def _score_two_plus_qubit_gates():
        # it is mandatory to have 1 and 2 qubit gates max
        more_than_two_qubit_gates = len(
            [
                op
                for op in response_circuit.all_operations()
                if cirq.num_qubits(op) > 2 or np.log2(len(cirq.unitary(op))) > 2
            ]
        )
        assert more_than_two_qubit_gates == 0, (
            f"Number of gates that need more than two "
            f"qubits: {more_than_two_qubit_gates} <-- it "
            f"should be zero!"
        )
        return 0, ""

    def _score_trace_distance():
        # extra points for exact equality
        u = response_unitary @ expected_unitary.conj().T
        trace_distance = cirq.trace_distance_from_angle_list(
            np.angle(np.linalg.eigvals(u))
        )
        assert (
            trace_distance < 1e-4
        ), f"trace distance of input.conj().T @ response is {trace_distance} > 1e-4"
        return max_trace_distance_score, ""

    def _score_circuit_structure():
        # the shorter your circuit/response the more points you get!
        num_two_qubit_gates = len(
            [op for op in response_circuit.all_operations() if cirq.num_qubits(op) == 2]
        )
        res_score = 0
        if num_two_qubit_gates == 0:
            res_score = max_two_qubit_gate_count_score
        elif num_two_qubit_gates >= generic_lower_bound:
            res_score = (
                generic_lower_bound
                / num_two_qubit_gates
                * max_two_qubit_gate_count_score
            )
        elif num_two_qubit_gates < generic_lower_bound:
            # bonus for going below the lower bound!
            res_score = (
                max_two_qubit_gate_count_score + max_two_qubit_gate_count_score / 2
            )
            result.msgs += " [WOW! 50% bonus] "

        extra_msgs = f"\n - 2-qubit gates in your result: {num_two_qubit_gates}"
        extra_msgs += f"\n - Lower bound for general case: {generic_lower_bound}"
        return res_score, extra_msgs

    def _score_sycamore():
        # extra points for compatibility with Sycamore
        cirq.Circuit(response, device=cirq.google.Sycamore)
        return max_sycamore_score, ""

    # running the scoring functions
    for task, score_func, fail_all_after, max_score in [
        ("2+ qubit gates", _score_two_plus_qubit_gates, True, 0),
        (
            f"Close in trace distance",
            _score_trace_distance,
            True,
            max_trace_distance_score,
        ),
        (
            "Circuit structure",
            _score_circuit_structure,
            False,
            max_two_qubit_gate_count_score,
        ),
        ("Valid for Sycamore device", _score_sycamore, False, max_sycamore_score),
    ]:
        result.msgs += f"\n{task} ({max_score} pts): "
        if skipped:
            result.msgs += "[skipped] "
        elif failed:
            result.msgs += f"✘\n"
        else:
            try:
                score, extra_msgs = score_func()
                result.msgs += f"✔ [{score} pts]"
                result.msgs += extra_msgs
                result.actual_score += score
            except BaseException as ex:
                if fail_all_after:
                    failed = True
                result.msgs += (
                    f"✘\n {type(ex)}: {str(ex)[:50]}" f"\n {traceback.format_exc()}"
                )
