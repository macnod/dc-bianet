<script>
	import { onMount } from 'svelte'

	$: connections = findConnections(selected)
 	export let elements
	export let selected

	function findConnections(selected) {
		return elements.filter(n => {
		  return n.group == 'edges' && n.data.source == selected;
		})
	}

	onMount(() => connections = findConnections(selected))

</script>

<h1>Node</h1>
<input bind:value={selected}/><br/>
Connections:
<div class="connections-pane">
<table class="connections">
	<tr>
		<th class="target-id-header">T</th>
		<th class="target-weight-header">W</th>
	</tr>
	{#each connections as cx}
		<tr>
			<td class="target-id-cell target-cell">{
				cx.data.target
		  }</td>
			<td class="target-weight-cell target-cell">{
				(Math.round(cx.data.weight * 100) / 100).toFixed(4)
		  }</td>
		</tr>
	{/each}
</table>
</div>
			
<style>
	.connections-pane {
		height: 440px;
		overflow-y: scroll;
	}

	.connections {
		width: 50%;
	}

	.target-cell {
		font-size: 75%;
		padding: 0px;
	}
	
	.target-id-cell {
		text-align: left;
	}

	.target-weight-cell {
	  text-align: right;
	}

	.target-id-header {
		text-align: left;
	}

	.target-weight-header {
	  text-align: right;
	}
</style>
