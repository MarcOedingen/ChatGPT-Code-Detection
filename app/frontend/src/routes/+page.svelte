<script lang="ts">
	import CodeEditor from '$lib/components/ui/codeEditor.svelte';
	import * as Form from "$lib/components/ui/form";
	import Label from '$lib/components/ui/label/label.svelte';
	import Progress from '$lib/components/ui/progress/progress.svelte';
	import {
		superForm
	} from 'sveltekit-superforms';
	import type { PageData } from './$types';
    
    export let data: PageData;
    let resultEl: HTMLDivElement;
    let prediction: 0 | 1 | null = null;
    let probability: number  = 0;

    const form = superForm(data.form, {
        dataType: "json",
        onResult: ({ result }) => {
            if (result.type !== "success") {
                return
            }
            // @ts-ignore
            const { data: { data: output} } = result;
            prediction = output.prediction;
            probability = output.probability;

            if (resultEl) {
                resultEl.scrollIntoView({ behavior: "smooth" });
            }
        },
        resetForm: false,
    
    });
    const { form: formData, enhance, delayed, message} = form;
</script>
    <div class="w-full mx-auto max-w-screen-xl p-4 flex items-center justify-between flex-col gap-5">
        <div class="w-full flex flex-col space-y-2">
            <form method="POST" use:enhance>
                <Form.Field {form} name="code">
                    <Form.Control let:attrs>
                    <Form.Label>Python Code Snippet</Form.Label>
                        <CodeEditor class="h-[20rem] border-2 overflow-y-auto " bind:value={$formData.code} />
                    </Form.Control>
                    <Form.FieldErrors />
                </Form.Field>
                <Form.Button loading={$delayed} class="float-right">Submit</Form.Button>
            </form>
        </div>

        <div class="w-full space-y-2">
            <Label class="text-left">Probability {#if probability}<span class="font-bold">{(probability * 100).toFixed(3)}%</span>{/if}</Label>
            <Progress max={1} value={probability} class="w-full" />
        </div>
        {#if $message !== undefined}
                    <div class="w-full bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
                        {$message}
                    </div>
                {/if}
        <div bind:this={resultEl} class="w-full">
            {#if prediction !== null}
                <div class="w-full bg-neutral-100 border border-neutral-400 text-neutral-700 px-4 py-3 rounded relative" role="alert">
                    {#if prediction === 1}
                        <span>🤖 The provided code snippet is likely to be generated <strong class="font-bold">with ChatGPT</strong></span>
                    {:else}
                    <span>✍️ The provided code snippet is likely to be <strong class="font-bold">not</strong> generated with ChatGPT</span>
                    {/if}
                </div>
            {/if}
        </div>
    </div>

