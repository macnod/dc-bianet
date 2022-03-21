<script>
	$: incoming = findIncoming(selected)
	$: outgoing = findOutgoing(selected)
	$: neuron = findNeuron(selected)

 	export let elements
	export let selected

	function findIncoming(selected) {
		return elements.filter(n => {
		  return n.group == 'edges' && n.data.target == selected;
		})
	}

	function findOutgoing(selected) {
		return elements.filter(n => {
		  return n.group == 'edges' && n.data.source == selected;
		})
	}

	function findNeuron(selected) {
		return elements.filter(n => {
			return n.group == 'nodes' && n.data.id == selected;
		})[0]
	}
</script>

<h1>Node</h1>
<input bind:value={selected}/><br/>
ID: {neuron.data.id}<br/>
<div class="connections-pane">
	<p> Incoming:</p>
	<table class="connections">
		<tr>
			<th class="target-id-header">S</th>
			<th class="target-id-header">T</th>
			<th class="target-weight-header">W</th>
 		</tr>
		{#each incoming as cx}
			<tr>
				<td class="target-id-cell target-cell">{cx.data.source}</td>
				<td class="target-id-cell target-cell">{cx.data.target}</td>
				<td class="target-weight-cell target-cell">{
					(Math.round(cx.data.weight * 100) / 100).toFixed(4)
					}</td>
			</tr>
		{/each}
	</table>
	<p>Outgoing:</p>
	<table class="connections">
		<tr>
			<th class="target-id-header">S</th>
			<th class="target-id-header">T</th>
			<th class="target-weight-header">W</th>
		</tr>
		{#each outgoing as cx}
			<tr>
				<td class="target-id-cell target-cell">{cx.data.source}</td>
				<td class="target-id-cell target-cell">{cx.data.target}</td>
				<td class="target-weight-cell target-cell">{
					(Math.round(cx.data.weight * 100) / 100).toFixed(4)
					}</td>
			</tr>
		{/each}
	</table>
</div>

<style>
	.connections-pane {
		height: 60%;
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
