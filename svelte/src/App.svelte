<script>
	import Graph from './Graph.svelte'
	import Neuron from './Neuron.svelte'
	import { createEventDispatcher } from 'svelte'

	const dispatch = createEventDispatcher();
	
	function getElements() {
		var elements = [];
		var layers = [128, 32, 8, 2];
		for (let layer = 0; layer < layers.length; layer++) {
			for (let neuronIndex = 0; neuronIndex < layers[layer]; neuronIndex++) {
				let neuronId = `${layer}:${neuronIndex}`;
				elements.push({group: 'nodes', data: {id: neuronId}});
			}
		}
		for (let layer = 0; layer < layers.length - 1; layer++) {
			for (let sourceIndex = 0; sourceIndex < layers[layer]; sourceIndex++) {
				let nextLayer = layer + 1;
				for (let targetIndex = 0; targetIndex < layers[nextLayer]; targetIndex++) {
					let sourceId = `${layer}:${sourceIndex}`;
					let targetId = `${nextLayer}:${targetIndex}`;
					elements.push({
						group: 'edges',
						data: {
							id: `${sourceId}->${targetId}`, 
							source: sourceId, 
							target: targetId,
							weight: Math.random(),
							layer: layer,
							inview: true,
							weak: false,
							selected: false
						}
					});
				}
			}
		}
		return elements;
	}

	function cullElements(elements) {
		
	}

  function handleSelected(event) {
		selected = event.detail.id;
		dispatch('selected', {id: selected})
	}

	let elements = getElements()
	let selected = '0:0'

</script>

<main>
	<h1>Network</h1>
	<div id="container">
		<div id="graph-container" class="child">
			<Graph elements={elements} on:selected={handleSelected}/>
		</div>
		<div id="neuron-container" class="child">
			<Neuron elements={elements} bind:selected={selected}/>
		</div>
	</div>
</main>

<style>
	h1 {
		color: #ff3e00;
		text-transform: uppercase;
		font-size: 4em;
		font-weight: 100;
	}

	#container {
		overflow: hidden;
		width: 100%;
	}

	.child {
		float: left;
	}

	#graph-container {
		width: 70%;
		height: 440px;
	}

	#neuron-container {
		width: 25%;
		padding: 10px;
		height: 400px;
	}

	@media (min-width: 640px) {
		main {
			max-width: none;
		}
	}
</style>
