<script>
	import Footer from '$lib/components/ui/footer.svelte';
import '../app.css';
</script>

<header class="relative z-50 w-full flex-none">
	<div class="w-full mx-auto max-w-screen-xl px-4">
		<div class="relative flex justify-center items-start py-[2.125rem] flex-col gap-4">
			<h1 class="">ChatGPT Code Detection</h1>
			<p>This web application serves as demonstration of the findings of the paper "ChatGPT Code Detection: Techniques for Uncovering the Source of Code".
				<br> The classification is done with a XGBoost model trained on this <a class="underline" target="_blank" href="https://th-koeln.sciebo.de/s/XZRR45yzO0rRuj3">dataset</a>.
				<br>It allows users to input Python code snippets and determine whether they were generated by ChatGPT or not.
				<br>For more information, please refer to the <a href="https://arxiv.org/abs/2405.15512" target="_blank" class="underline">paper</a>.
			</p>
		</div>
	</div>
</header>

<main class="mx-auto w-screen flex-col justify-center pt-2 ">
	<slot />
	<Footer />
</main>
