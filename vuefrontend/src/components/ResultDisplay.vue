<template>
  <div v-if="sections && sections.length != 0">
    <p>{{ this.video_labels }}</p>
    <li v-for="idx of section_labels.length" :key="idx">{{ sections[idx - 1] }} ~ {{ sections[idx] }}: {{ section_labels[idx - 1] }}</li>
  </div>
</template>

<script>
export default {
  name: "ResultDisplay",
  props: {
    jsonData: {
      type: Object
    }
  },
  data () {
    return {
      sections: undefined,
      section_labels: undefined,
      video_labels: undefined
    }
  },
  beforeUpdate() {
    this.sections = this.jsonData.sections
    this.section_labels = this.jsonData.section_labels
    this.video_labels = this.jsonData.video_labels
  },
}
</script>

<style>

</style>