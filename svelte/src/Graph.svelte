<script>
	import cytoscape from 'cytoscape'
	import { onMount } from 'svelte'
	import { createEventDispatcher } from 'svelte'

	export let elements

	const dispatch = createEventDispatcher();
	
	var style = [ // the stylesheet for the graph
    {
      selector: 'node',
      style: {
        'background-color': '#000'
      }
    },

    {
      selector: 'edge',
      style: {
        'width': 1,
        'line-color': '#888',
        'target-arrow-color': '#000',
        'target-arrow-shape': 'triangle',
        'curve-style': 'haystack'
      }
    }
  ]

	let selected
	let graphdiv
	let cy
	let edgesVisible = true
	$: toggleEdgesText = edgesVisible ? "Hide" : "Show"
	let edges = []
	let notInView = []
	let unselectedEdges = []

	onMount( () => {
		cy = cytoscape({
			container: graphdiv,
			elements: elements,
			style: style,
			layout: {
				name: 'breadthfirst',
				directed: true,
				avoidOverlap: true
			},
			zoom: 1
		})
		toggleEdges()
		cy.on('viewport', hideNodes)
		cy.on('tap', function(event) {
			selected = event.target.data().id
			dispatch('selected', {id: selected})
		})
	})

	function toggleEdges() {
		edgesVisible = !edgesVisible
		if (edgesVisible) {
			if (edges) {
				edges.restore()
			}
		} else {
			edges = cy.remove('edges')
		}
	}

	function nodesInView() {
		const extent = cy.extent()
		return cy.nodes().filter(n => {
			const bb = n.boundingBox()
			return bb.x1 > ext.x1 && bb.x2 < ext.x2 && bb.y1 > ext.y1 && bb.y2 < ext.y2
		})
	}

	function markNotInView() {
		const view = cy.extent()
		cy.nodes().data('inview', true)
		cy.nodes().filter(n => {
			const p = n.position()
			return p.x < view.x1 || p.x > view.x2 || p.y < view.y1 || p.y > view.y2
		}).data('inview', false)
	}

	function markWeak() {
		n = cy.nodes().length
		m = n <= 128 ? 1 : Math.floor(n / 128)
		cy.nodes().data('weak', false)
		cy.nodes().filter(id => {return id % n == 0}).data('weak', true)
	}

	function hideNodes() {
		if (notInView.length > 0) {
			notInView.restore()
		}
		markNotInView()
		notInView = cy.remove('[!inview]')
	}

</script>

<h1>Graph</h1>
<button on:click={toggleEdges}>{toggleEdgesText} Edges</button>
<div id="graph" bind:this={graphdiv}></div>
<style>
	#graph {
	  width: 100%;
		height: 300px;
		display: block;
		margin: 0 auto;
		border: 1px solid black;
	}
</style>
