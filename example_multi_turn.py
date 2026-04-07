"""
Example: Multi-turn Medical Dialogue Generation with ToM Constraints

This script demonstrates how to use the new phased generation approach:
1. Phase 1: Generate metadata + system prompt + initial patient complaint
2. Phase 2-N: Iteratively generate each turn with ToM reasoning
"""

from pathlib import Path
from multi_turn_generator import MultiTurnGenerator


def example_basic_usage() -> None:
    """Basic example with default settings"""

    print("=" * 80)
    print("Example 1: Basic Multi-Turn Generation")
    print("=" * 80)
    print()

    generator = MultiTurnGenerator(
        turns=4,
        model="qwen-plus",
    )

    results = generator.run(
        file_path=Path("ehr_bench_decision_making.jsonl"),
        output_path=Path("output_example1.jsonl"),
        max_samples=3,
    )

    print(f"\n✓ Successfully generated {len(results)} multi-turn dialogues")


def example_custom_turns() -> None:
    """Example with custom number of turns"""

    print("\n" + "=" * 80)
    print("Example 2: Custom Number of Turns (6 turns)")
    print("=" * 80)
    print()

    generator = MultiTurnGenerator(
        turns=6,
        model="qwen-plus",
    )

    results = generator.run(
        file_path=Path("ehr_bench_decision_making.jsonl"),
        output_path=Path("output_example2.jsonl"),
        max_samples=1,
    )

    if results:
        sample = results[0]
        dialogue = sample.get("prompt", [])
        assistant_turns = [m for m in dialogue if m["role"] == "assistant"]
        print(f"\n✓ Generated dialogue with {len(assistant_turns)} assistant turns")
        print(f"  Total messages: {len(dialogue)}")


def example_single_sample_detailed() -> None:
    """Detailed example showing the generation process for a single sample"""

    print("\n" + "=" * 80)
    print("Example 3: Detailed Single Sample Generation")
    print("=" * 80)
    print()

    from prompt import MedicalToMPromptGenerator

    file_path = Path("ehr_bench_decision_making.jsonl")
    generator = MedicalToMPromptGenerator(turns=4)
    generator.read_datas(file_path)

    if generator.datas and len(generator.datas) > 0:
        sample_data = generator.datas[0]

        multi_gen = MultiTurnGenerator(turns=4, model="qwen-plus")

        print(f"Processing sample {sample_data.get('idx', 0)}...")
        print("-" * 80)

        result = multi_gen.generate_single_sample(sample_data)

        print("\n" + "-" * 80)
        print("Generated Result Structure:")
        print("-" * 80)
        print(json.dumps({
            k: v for k, v in result.items()
            if k != 'prompt'
        }, indent=2, ensure_ascii=False))

        print("\nDialogue Preview:")
        for i, msg in enumerate(result['prompt'][:6]):
            role = msg['role'].upper()
            content_preview = msg['content'][:150].replace('\n', ' ')
            print(f"\n[{i+1}] {role}:")
            print(f"    {content_preview}...")

        total_msgs = len(result['prompt'])
        if total_msgs > 6:
            print(f"\n... and {total_msgs - 6} more messages")

        assistant_turns = [m for m in result['prompt'] if m['role'] == 'assistant']
        print(f"\nTotal: {total_msgs} messages ({len(assistant_turns)} assistant turns)")


import json


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("# Multi-Turn Medical Dialogue Generation Examples")
    print("# Using Phased Approach with ToM Constraints")
    print("#" * 80 + "\n")

    try:
        example_basic_usage()
        # Uncomment to run other examples:
        # example_custom_turns()
        # example_single_sample_detailed()
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nMake sure you have:")
        print("1. Set up DASHSCOPE_API_KEY in .env file")
        print("2. Installed required dependencies: openai, python-dotenv")
        print("3. The ehr_bench_decision_making.jsonl file exists")
