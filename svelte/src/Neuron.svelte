<script>
	$: incoming = findIncoming(selected)
	$: outgoing = findOutgoing(selected)
	$: neuron = findNeuron(selected)

 	export let elements
	export let selected

	function findIncoming(selected) {
		if (selected) {
			return elements.filter(n => {
				return n.group == 'edges' && n.data.target == selected;
			})
		}
	}

	function findOutgoing(selected) {
		if (selected) {
			return elements.filter(n => {
				return n.group == 'edges' && n.data.source == selected;
			})
		}
	}

	function findNeuron(selected) {
		if (selected) {
			return elements.filter(n => {
				return n.group == 'nodes' && n.data.id == selected;
			})[0]
		}
	}
</script>

<h1>Node</h1>
<table>
	<tr><td>ID</td><td><input bind:value={selected}/></td></tr>
	{#if incoming.length > 0 || outgoing.length > 0}
		<tr><td>In➝Out</td><td>{incoming.length}➝{outgoing.length}</td></tr>
	{/if}
</table>
<div class="connections-pane">
	{#if incoming.length > 0 || outgoing.length > 0}
		<table class="connections">
			<tr>
				<th class="target-id-header">S</th>
				<th class="arrow">&nbsp</th>
				<th class="target-id-header">T</th>
				<th class="target-weight-header">W</th>
				<th class="target-mom-header">M</th>
			</tr>
			{#each incoming as cx}
				<tr>
					<td class="target-id-cell target-cell">{cx.data.source}</td>
					<td class="arrow">➝</td>
					<td class="target-id-cell target-cell">{cx.data.target}</td>
					<td class="target-weight-cell target-cell">{
						(Math.round(cx.data.weight * 100) / 100).toFixed(4)
						}</td>
					<td class="target-momentum-cell target-cell">0.0000</td>
				</tr>
			{/each}
			{#each outgoing as cx}
				<tr>
					<td class="target-id-cell target-cell">{cx.data.source}</td>
					<td class="arrow">➝</td>
					<td class="target-id-cell target-cell">{cx.data.target}</td>
					<td class="target-weight-cell target-cell">{
						(Math.round(cx.data.weight * 100) / 100).toFixed(4)
						}</td>
					<td class="target-momentum-cell target-cell">0.0000</td>
				</tr>
			{/each}
		</table>
	{/if}
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
		padding-left: 2px;
		padding-right: 0px;
		padding-top: 0px;
		padding-bottom: 0px;
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
