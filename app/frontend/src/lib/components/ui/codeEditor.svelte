<!-- TODO: add python lang support -->

<script lang="ts">
    import CodeMirror from "svelte-codemirror-editor";

    export let value: string;
</script>

<div {...$$restProps}><CodeMirror bind:value  /></div>

