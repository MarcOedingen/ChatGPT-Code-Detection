<script lang="ts">
	import * as Button from "$lib/components/ui/button/index.js";
	import { LoaderCircle } from 'lucide-svelte';

	type $$Props = Button.Props;
	type $$Events = Button.Events;

	export let loading = false;

</script>

<Button.Root disabled={loading} type="submit" on:click on:keydown {...$$restProps}>
	{#if loading}
		<LoaderCircle class="h-4 w-4 animate-spin" />
	{:else}
		<slot />
	{/if}
</Button.Root>
